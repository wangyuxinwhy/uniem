import functools
import logging
import os
from pathlib import Path
from typing import Sequence, cast

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import Dataset as HFDataset
from datasets import DatasetDict as HFDatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # type: ignore

from uniem.data import (
    FinetuneDataset,
    PairCollator,
    PrefixFinetuneDataset,
    ScoredPairCollator,
    TripletCollator,
)
from uniem.data_structures import RecordType, get_record_type
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
from uniem.utils import create_adamw_optimizer, split_dataset_dict

logger = logging.getLogger(__name__)
RawDataset = Sequence[dict] | dict[str, Sequence[dict]] | HFDatasetDict | HFDataset


class FineTuner:
    def __init__(
        self,
        model_name_or_path: str,
        dataset: RawDataset,
    ):
        self.model_name_or_path = model_name_or_path
        self.raw_dataset = dataset
        if isinstance(self.raw_dataset, dict):
            (
                self.raw_train_dataset,
                self.raw_validation_dataset,
            ) = split_dataset_dict(self.raw_dataset)
        else:
            self.raw_train_dataset, self.raw_validation_dataset = (
                self.raw_dataset,
                None,
            )
        self.record_type = get_record_type(self.raw_train_dataset[0])  # type: ignore
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def create_finetune_datasets(
        self,
    ) -> tuple[FinetuneDataset, FinetuneDataset | None]:
        train_dataset = FinetuneDataset(self.raw_train_dataset)
        validation_dataset = FinetuneDataset(self.raw_validation_dataset) if self.raw_validation_dataset is not None else None
        return train_dataset, validation_dataset

    def create_dataloaders(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        drop_last: bool = False,
        max_length: int | None = None,
    ) -> tuple[DataLoader, DataLoader | None]:
        train_dataset, validation_dataset = self.create_finetune_datasets()

        match self.record_type:
            case RecordType.PAIR:
                data_collator = PairCollator(tokenizer=self.tokenizer, max_length=max_length)
            case RecordType.TRIPLET:
                data_collator = TripletCollator(tokenizer=self.tokenizer, max_length=max_length)
            case RecordType.SCORED_PAIR:
                data_collator = ScoredPairCollator(tokenizer=self.tokenizer, max_length=max_length)

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
        temperature: float | None = None,
        embedding_strategy: PoolingStrategy = PoolingStrategy.last_mean,
    ) -> EmbedderForTrain:
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
        return model

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
            project_dir=str(output_dir),
            automatic_checkpoint_naming=True,
            total_limit=num_max_checkpoints,
        )
        accelerator = Accelerator(
            mixed_precision=mixed_precision.value,
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_config=project_config,
            log_with=['tensorboard'] if use_tensorboard else None,
        )
        accelerator.init_trackers('uniem')

        set_seed(seed)
        accelerator.print(f'Start with seed: {seed}')
        accelerator.print(f'Output dir: {output_dir}')

        train_dataloader, validation_dataloader = self.create_dataloaders(
            batch_size=batch_size,
            drop_last=drop_last,
            max_length=max_length,
            num_workers=num_workers,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        validation_dataloader = accelerator.prepare(validation_dataloader) if validation_dataloader is not None else None

        model = self.create_finetune_model(temperature=temperature, embedding_strategy=embedding_strategy)
        model.embedder.encoder.config.pad_token_id = self.tokenizer.pad_token_id
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
        self.tokenizer.save_pretrained(output_dir / 'model')


def partial_freeze_gradients(grad, train_indices: torch.Tensor):
    train_indices_grad = grad[train_indices, :]
    grad.zero_()
    grad[train_indices, :] = train_indices_grad
    return grad


class PrefixFineTuner(FineTuner):
    def __init__(
        self,
        model_name_or_path: str,
        dataset: RawDataset,
        additional_special_tokens: list[str],
        prefix: str | None = None,
        only_train_additional_special_tokens: bool = True,
    ):
        super().__init__(model_name_or_path, dataset)
        self.special_prefix_tokens = additional_special_tokens
        self.prefix = ''.join(self.special_prefix_tokens) if prefix is None else prefix
        self.tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})  # type: ignore
        self.additional_special_token_ids = self.tokenizer.convert_tokens_to_ids(additional_special_tokens)
        self.only_train_additional_special_tokens = only_train_additional_special_tokens

    def create_finetune_datasets(
        self,
    ) -> tuple[FinetuneDataset, FinetuneDataset | None]:
        train_dataset = PrefixFinetuneDataset(self.raw_train_dataset, prefix=self.prefix)
        validation_dataset = (
            PrefixFinetuneDataset(self.raw_validation_dataset, prefix=self.prefix)
            if self.raw_validation_dataset is not None
            else None
        )
        return train_dataset, validation_dataset

    def create_finetune_model(
        self,
        temperature: float | None = None,
        embedding_strategy: PoolingStrategy = PoolingStrategy.last_mean,
    ) -> EmbedderForTrain:
        model = super().create_finetune_model(temperature, embedding_strategy)
        model.embedder.encoder.resize_token_embeddings(len(self.tokenizer))
        hook = functools.partial(
            partial_freeze_gradients,
            train_indices=torch.tensor(self.additional_special_token_ids),
        )
        if self.only_train_additional_special_tokens:
            for param in model.parameters():
                param.requires_grad = False
            embedding_layer_weight = model.embedder.encoder.get_input_embeddings().weight
            embedding_layer_weight = cast(torch.nn.Parameter, embedding_layer_weight)
            embedding_layer_weight.requires_grad = True
            embedding_layer_weight.register_hook(hook)
        return model

    def run(
        self,
        temperature: float | None = None,
        embedding_strategy: PoolingStrategy = PoolingStrategy.last_mean,
        batch_size: int = 32,
        drop_last: bool = True,
        max_length: int = 512,
        lr: float = 1e-2,
        epochs: int = 5,
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
            project_dir=str(output_dir),
            automatic_checkpoint_naming=True,
            total_limit=num_max_checkpoints,
        )
        accelerator = Accelerator(
            mixed_precision=mixed_precision.value,
            gradient_accumulation_steps=gradient_accumulation_steps,
            project_config=project_config,
            log_with=['tensorboard'] if use_tensorboard else None,
        )
        accelerator.init_trackers('uniem')

        set_seed(seed)
        accelerator.print(f'Start with seed: {seed}')
        accelerator.print(f'Output dir: {output_dir}')

        train_dataloader, validation_dataloader = self.create_dataloaders(
            batch_size=batch_size,
            drop_last=drop_last,
            max_length=max_length,
            num_workers=num_workers,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        validation_dataloader = accelerator.prepare(validation_dataloader) if validation_dataloader is not None else None

        model = self.create_finetune_model(temperature=temperature, embedding_strategy=embedding_strategy)
        model.embedder.encoder.config.pad_token_id = self.tokenizer.pad_token_id
        model = accelerator.prepare(model)

        # Optimizer & LRScheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = accelerator.prepare(optimizer)

        # Trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            accelerator=accelerator,
            epochs=epochs,
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
