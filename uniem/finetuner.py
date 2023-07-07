import logging
import os
from pathlib import Path
from typing import Sequence, cast

from accelerate import Accelerator
from accelerate.tracking import GeneralTracker
from accelerate.utils import LoggerType, ProjectConfiguration, set_seed
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # type: ignore

from uniem.data import (
    FinetuneDataset,
    PairCollator,
    ScoredPairCollator,
    TripletCollator,
)
from uniem.data_structures import RecordType, infer_record_type
from uniem.model import (
    EmbedderForPairInBatchNegTrain,
    EmbedderForScoredPairTrain,
    EmbedderForTrain,
    EmbedderForTripletInBatchNegTrain,
    InBatchNegLossType,
    PoolingStrategy,
)
from uniem.trainer import Trainer
from uniem.training_strategy import FullParametersTraining, PrefixTraining, TrainingStrategy
from uniem.types import MixedPrecisionType
from uniem.utils import create_adamw_optimizer, find_executable_batch_size, split_dataset_dict

logger = logging.getLogger(__name__)
RawDataset = Sequence[dict] | dict[str, Sequence[dict]] | HFDatasetDict | HFDataset


def suggest_lr(model_name: str) -> float:
    default_lr = 3e-5
    if 'm3e-small' in model_name:
        lr = 1e-4
    elif 'm3e-base' in model_name:
        lr = 5e-5
    elif 'm3e-large' in model_name:
        lr = 8e-6
    else:
        lr = default_lr
    logger.info(f'Suggested learning rate: {lr}')
    return lr


class FineTuner:
    def __init__(
        self,
        model_name_or_path: str,
        dataset: RawDataset,
        model_class: str | None = None,
        record_type: RecordType | str | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.raw_dataset = dataset
        if isinstance(self.raw_dataset, dict):
            (
                self.raw_train_dataset,
                self.raw_validation_dataset,
            ) = split_dataset_dict(self.raw_dataset)
        else:
            self.raw_train_dataset = self.raw_dataset
            self.raw_validation_dataset = None

        record_type = RecordType(record_type) if isinstance(record_type, str) else record_type
        self.record_type = record_type or infer_record_type(self.raw_train_dataset[0])
        self.model_class = model_class
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def create_finetune_datasets(
        self,
    ) -> tuple[FinetuneDataset, FinetuneDataset | None]:
        train_dataset = FinetuneDataset(self.raw_train_dataset)
        validation_dataset = FinetuneDataset(self.raw_validation_dataset) if self.raw_validation_dataset is not None else None
        return train_dataset, validation_dataset

    def create_dataloaders(
        self,
        train_dataset: FinetuneDataset,
        validation_dataset: FinetuneDataset | None,
        batch_size: int = 64,
        num_workers: int = 0,
        drop_last: bool = False,
        max_length: int | None = None,
    ) -> tuple[DataLoader, DataLoader | None]:

        match self.record_type:
            case RecordType.PAIR:
                data_collator = PairCollator(tokenizer=self.tokenizer, max_length=max_length)
            case RecordType.TRIPLET:
                data_collator = TripletCollator(tokenizer=self.tokenizer, max_length=max_length)
            case RecordType.SCORED_PAIR:
                data_collator = ScoredPairCollator(tokenizer=self.tokenizer, max_length=max_length)
            case _:
                raise ValueError('Only supports pair, triplet and scored pair record.')

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=drop_last,
        )

        if validation_dataset is not None:
            validation_dataloader = DataLoader(
                validation_dataset,
                batch_size=batch_size,
                collate_fn=data_collator,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            validation_dataloader = None
        return train_dataloader, validation_dataloader

    def create_finetune_model(
        self,
        temperature: float = 0.05,
        pooling_strategy: PoolingStrategy = PoolingStrategy.last_mean,
        model_class: str | None = None,
    ) -> EmbedderForTrain:
        match self.record_type:
            case RecordType.PAIR:
                model = EmbedderForPairInBatchNegTrain(
                    model_name_or_path=self.model_name_or_path,
                    model_class=model_class,
                    temperature=temperature,
                    loss_type=InBatchNegLossType.softmax,
                    pooling_strategy=pooling_strategy,
                )
            case RecordType.TRIPLET:
                model = EmbedderForTripletInBatchNegTrain(
                    model_name_or_path=self.model_name_or_path,
                    model_class=model_class,
                    temperature=temperature,
                    loss_type=InBatchNegLossType.softmax,
                    pooling_strategy=pooling_strategy,
                )
            case RecordType.SCORED_PAIR:
                model = EmbedderForScoredPairTrain(
                    model_name_or_path=self.model_name_or_path,
                    model_class=model_class,
                    temperature=temperature,
                    pooling_strategy=pooling_strategy,
                )
        return model

    @find_executable_batch_size(starting_batch_size=256)
    def run(
        self,
        # Model
        temperature: float = 0.05,
        pooling_strategy: PoolingStrategy = PoolingStrategy.last_mean,
        training_strategy: TrainingStrategy = FullParametersTraining(),
        # Optimizer
        lr: float | None = None,
        weight_decay: float = 1e-3,
        num_warmup_steps: float = 0.05,
        # Data
        batch_size: int = 256,
        max_length: int = 512,
        drop_last: bool = False,
        # Trainer
        epochs: int = 3,
        mixed_precision: MixedPrecisionType = MixedPrecisionType.no,
        gradient_accumulation_steps: int = 1,
        save_on_epoch_end: bool = False,
        num_max_checkpoints: int = 1,
        log_with: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,
        num_workers: int = 0,
        seed: int = 42,
        output_dir: Path | str | None = None,
    ):

        os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
        if num_workers >= 1:
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

        output_dir = Path(output_dir) if output_dir is not None else Path('finetuned-model')
        project_config = ProjectConfiguration(
            project_dir=str(output_dir),
            automatic_checkpoint_naming=True,
            total_limit=num_max_checkpoints,
        )
        accelerator = Accelerator(
            mixed_precision=mixed_precision.value,
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_config=project_config,
            log_with=log_with,
        )
        accelerator.init_trackers('uniem')

        set_seed(seed)
        accelerator.print(f'Batch size: {batch_size}')
        accelerator.print(f'Start with seed: {seed}')
        accelerator.print(f'Output dir: {output_dir}')

        train_dataset, validation_dataset = self.create_finetune_datasets()
        if isinstance(training_strategy, PrefixTraining):
            self.tokenizer = training_strategy.apply_tokenizer(self.tokenizer)
            train_dataset = training_strategy.apply_dataset(train_dataset)
            if validation_dataset:
                validation_dataset = training_strategy.apply_dataset(validation_dataset)

        train_dataloader, validation_dataloader = self.create_dataloaders(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            max_length=max_length,
            num_workers=num_workers,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        validation_dataloader = accelerator.prepare(validation_dataloader) if validation_dataloader is not None else None

        model = self.create_finetune_model(
            temperature=temperature, pooling_strategy=pooling_strategy, model_class=self.model_class
        )
        model = training_strategy.apply_model(model)
        model.embedder.encoder.config.pad_token_id = self.tokenizer.pad_token_id
        model = accelerator.prepare(model)

        # Optimizer & LRScheduler
        lr = lr or suggest_lr(self.model_name_or_path)
        optimizer = create_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
        total_steps = len(train_dataloader) * epochs
        if num_warmup_steps <= 0:
            lr_scheduler = None
        else:
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
        self.tokenizer.save_pretrained(output_dir / 'model')
