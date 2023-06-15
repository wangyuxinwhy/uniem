from dataclasses import fields
import os
import logging
from pathlib import Path
from typing import Sequence, cast

from datasets import DatasetDict, Dataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from uniem.data import PairCollator, ScoredPairCollator, TripletCollator, FinetuneDataset
from uniem.data_structures import RecordType, record_type_cls_map
from uniem.model import (
    EmbedderForPairInBatchNegTrain,
    EmbedderForScoredPairTrain,
    EmbedderForTrain,
    EmbedderForTripletInBatchNegTrain,
    InBatchNegLossType,
    PoolingStrategy,
)
from uniem.trainer import Trainer
from uniem.types import MixedPrecisionType
from uniem.utils import create_adamw_optimizer


logger = logging.getLogger(__name__)


class FineTuner:
    def __init__(
        self,
        model_name_or_path: str,
        dataset: Sequence[dict] | dict[str, Sequence[dict]] | DatasetDict | Dataset,
    ):
        self.model_name_or_path = model_name_or_path

        self.raw_dataset = dataset
        if isinstance(dataset, dict):
            train_dataset = dataset['train']
            if 'dev' in dataset:
                validation_dataset = dataset['dev']
            elif 'validation' in dataset:
                validation_dataset = dataset['validation']
            else:
                logger.warning(
                    'No validation dataset found in dataset_dict, validation dataset key should be either "dev" or "validation"'
                )
                validation_dataset = None
        else:
            train_dataset = dataset
            validation_dataset = None

        self.record_type = self.get_record_type(train_dataset[0])
        self.train_dataset = FinetuneDataset(train_dataset, self.record_type)
        if validation_dataset is not None:
            self.validation_dataset = FinetuneDataset(validation_dataset, self.record_type)
        else:
            self.validation_dataset = None

    def get_record_type(self, record: dict) -> RecordType:
        record_type_field_names_map = {
            record_type: [field.name for field in fields(record_cls)] for record_type, record_cls in record_type_cls_map.items()
        }
        for record_type, field_names in record_type_field_names_map.items():
            if all(field_name in record for field_name in field_names):
                return record_type
        raise ValueError(f'Unknown record type, record: {record}')

    def run(
        self,
        temperature: float | None = None,
        embedding_strategy: PoolingStrategy = PoolingStrategy.last_mean,
        batch_size: int = 32,
        drop_last: bool = True,
        max_length: int = 512,
        lr: float = 3e-5,
        weight_decay: float = 1e-3,
        num_warmup_steps: float = 0.05,
        epochs: int = 3,
        mixed_precision: MixedPrecisionType = MixedPrecisionType.no,
        gradient_accumulation_steps: int = 1,
        save_on_epoch_end: bool = False,
        num_max_checkpoints: int = 1,
        use_tensorboard: bool = False,
        num_workers: int = 0,
        seed: int = 42,
        output_dir: Path | str | None = None,
    ):
        os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
        if num_workers >= 1:
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

        output_dir = Path(output_dir) if output_dir is not None else Path('finetuned-model')
        project_config = ProjectConfiguration(
            project_dir=str(output_dir), automatic_checkpoint_naming=True, total_limit=num_max_checkpoints
        )
        accelerator = Accelerator(
            mixed_precision=mixed_precision.value,
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_config=project_config,
            log_with=['tensorboard'] if use_tensorboard else None,
        )
        accelerator.init_trackers('m3e')

        set_seed(seed)
        accelerator.print(f'Start with seed: {seed}')
        accelerator.print(f'Output dir: {output_dir}')

        # DataLoader
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        match self.record_type:
            case RecordType.PAIR:
                data_collator = PairCollator(tokenizer=tokenizer, max_length=max_length)
            case RecordType.TRIPLET:
                data_collator = TripletCollator(tokenizer=tokenizer, max_length=max_length)
            case RecordType.SCORED_PAIR:
                data_collator = ScoredPairCollator(tokenizer=tokenizer, max_length=max_length)

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )
        train_dataloader = accelerator.prepare(train_dataloader)

        if self.validation_dataset is not None:
            validation_dataloader = DataLoader(
                self.validation_dataset,
                batch_size=batch_size,
                collate_fn=data_collator,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
            validation_dataloader = accelerator.prepare(validation_dataloader)
        else:
            validation_dataloader = None

        match self.record_type:
            case RecordType.PAIR:
                model = EmbedderForPairInBatchNegTrain(
                    model_name_or_path=self.model_name_or_path,
                    temperature=temperature,
                    loss_type=InBatchNegLossType.softmax,
                    embedding_strategy=embedding_strategy,
                )
            case RecordType.TRIPLET:
                model = EmbedderForTripletInBatchNegTrain(
                    model_name_or_path=self.model_name_or_path,
                    temperature=temperature,
                    loss_type=InBatchNegLossType.softmax,
                    embedding_strategy=embedding_strategy,
                )
            case RecordType.SCORED_PAIR:
                model = EmbedderForScoredPairTrain(
                    model_name_or_path=self.model_name_or_path,
                    temperature=temperature,
                    embedding_strategy=embedding_strategy,
                )

        model.embedder.encoder.config.pad_token_id = tokenizer.pad_token_id
        model = accelerator.prepare(model)

        # Optimizer & LRScheduler
        optimizer = create_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
        total_steps = len(train_dataloader) * epochs
        if num_warmup_steps < 1:
            num_warmup_steps = int(num_warmup_steps * total_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(num_warmup_steps),
            num_training_steps=total_steps,
        )
        optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

        # Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            accelerator=accelerator,
            epochs=epochs,
            lr_scheduler=lr_scheduler,
            log_interval=10,
            save_on_epoch_end=save_on_epoch_end,
        )
        accelerator.print(f'Start training for {epochs} epochs')
        trainer.train()

        accelerator.wait_for_everyone()
        accelerator.print('Training finished')

        accelerator.print('Saving model')
        unwrapped_model = cast(EmbedderForTrain, accelerator.unwrap_model(model))

        unwrapped_model.embedder.save_pretrained(output_dir / 'model')
        tokenizer.save_pretrained(output_dir / 'model')
