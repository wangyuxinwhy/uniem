import logging
import os
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence, Sized, cast

import torch
from accelerate import Accelerator
from accelerate.tracking import GeneralTracker
from accelerate.utils import LoggerType, ProjectConfiguration, set_seed
from datasets import Dataset as HFDataset
from datasets import IterableDataset as HFIterableDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # type: ignore

from uniem.data import (
    FinetuneDataset,
    FinetuneIterableDataset,
    PairCollator,
    ScoredPairCollator,
    TripletCollator,
)
from uniem.data_structures import RecordType, infer_record_type
from uniem.model import (
    Embedder,
    EmbedderForPairInBatchNegTrain,
    EmbedderForScoredPairTrain,
    EmbedderForTrain,
    EmbedderForTripletInBatchNegTrain,
    InBatchNegLossType,
    UniemEmbedder,
    create_uniem_embedder,
)
from uniem.trainer import Trainer
from uniem.training_strategy import FullParametersTraining, PrefixTraining, TrainingStrategy
from uniem.types import MixedPrecisionType, Tokenizer
from uniem.utils import create_adamw_optimizer, find_executable_batch_size, split_dataset_dict

logger = logging.getLogger(__name__)
MapStyleDataset = Sequence[dict] | HFDataset
IterableStyleDataset = Iterable[dict] | HFIterableDataset
SupportedDataset = MapStyleDataset | IterableStyleDataset
SupportedDatasetDict = dict[str, SupportedDataset]


class ModelType(str, Enum):
    uniem = 'uniem'
    text2vec = 'text2vec'
    sentence_transformers = 'sentence_transformers'
    huggingface = 'huggingface'
    custom = 'custom'


def suggest_lr(model: torch.nn.Module) -> float:
    num_training_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if num_training_params <= 80_000_000:
        return 1e-4
    elif num_training_params <= 200_000_000:
        return 5e-5
    else:
        return 8e-6


class FineTuner:
    accelerator: Accelerator

    def __init__(
        self,
        embedder: Embedder,
        tokenizer: Tokenizer,
        dataset: SupportedDatasetDict | SupportedDataset,
        model_type: ModelType | str = ModelType.uniem,
        record_type: RecordType | str | None = None,
    ):
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.raw_dataset = dataset
        if isinstance(self.raw_dataset, dict):
            raw_dataset = cast(SupportedDatasetDict, self.raw_dataset)
            (
                self.raw_train_dataset,
                self.raw_validation_dataset,
            ) = split_dataset_dict(raw_dataset)
        else:
            self.raw_train_dataset = self.raw_dataset
            self.raw_validation_dataset = None

        record_type = RecordType(record_type) if isinstance(record_type, str) else record_type
        self.record_type = record_type or infer_record_type(next(iter(self.raw_train_dataset)))
        self.model_type = ModelType(model_type)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dataset: SupportedDatasetDict | SupportedDataset,
        model_type: ModelType | str | None = None,
        record_type: RecordType | str | None = None,
    ):
        if model_type is None:
            if 'sentence-transformers' in model_name_or_path:
                model_type = ModelType.sentence_transformers
            elif 'text2vec' in model_name_or_path:
                model_type = ModelType.text2vec
            elif 'm3e' in model_name_or_path:
                model_type = ModelType.uniem
            else:
                model_type = ModelType.huggingface
            logger.info(f'Auto detect model type: {model_type}')
        else:
            model_type = ModelType(model_type)

        match model_type:
            case ModelType.uniem:
                embedder = UniemEmbedder.from_pretrained(model_name_or_path)
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            case ModelType.huggingface | ModelType.text2vec:
                embedder = create_uniem_embedder(model_name_or_path)
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            case ModelType.sentence_transformers:
                try:
                    from uniem.integration.sentence_transformers_wrapper import SentenceTransformerWrapper

                    embedder = SentenceTransformerWrapper(model_name_or_path)   # type: ignore
                    tokenizer = embedder.tokenizer
                    tokenizer = cast(Tokenizer, tokenizer)
                except ImportError:
                    raise ImportError('can not find sentence_transformers, pip install sentence_transformers')
            case ModelType.custom:
                raise ValueError('model_type is custom, you should create embedder by yourself')

        return cls(embedder=embedder, tokenizer=tokenizer, dataset=dataset, record_type=record_type, model_type=model_type)

    def create_finetune_datasets(
        self,
    ) -> tuple[FinetuneDataset | FinetuneIterableDataset, FinetuneDataset | FinetuneIterableDataset | None]:
        if not isinstance(self.raw_train_dataset, Sized):
            raw_train_dataset = cast(IterableStyleDataset, self.raw_train_dataset)
            train_dataset = FinetuneIterableDataset(raw_train_dataset, record_type=self.record_type)
        else:
            train_dataset = FinetuneDataset(self.raw_train_dataset, record_type=self.record_type)

        if self.raw_validation_dataset is None:
            validation_dataset = None
        elif not isinstance(self.raw_validation_dataset, Sized):
            raw_validation_dataset = cast(IterableStyleDataset, self.raw_validation_dataset)
            validation_dataset = FinetuneIterableDataset(raw_validation_dataset, record_type=self.record_type)
        else:
            validation_dataset = FinetuneDataset(self.raw_validation_dataset, record_type=self.record_type)

        return train_dataset, validation_dataset

    def create_dataloaders(
        self,
        train_dataset: FinetuneDataset | FinetuneIterableDataset,
        validation_dataset: FinetuneDataset | FinetuneIterableDataset | None,
        batch_size: int = 64,
        num_workers: int = 0,
        drop_last: bool = False,
        shuffle: bool = False,
        max_length: int | None = None,
    ) -> tuple[DataLoader, DataLoader | None]:

        match self.record_type:
            case RecordType.PAIR:
                data_collator = PairCollator(tokenizer=self.tokenizer, max_length=max_length)
            case RecordType.TRIPLET:
                data_collator = TripletCollator(tokenizer=self.tokenizer, max_length=max_length)
            case RecordType.SCORED_PAIR:
                data_collator = ScoredPairCollator(tokenizer=self.tokenizer, max_length=max_length)

        if not isinstance(train_dataset, Sized) and shuffle:
            shuffle = False
            self.accelerator.print('Disable shuffle for iterable dataset')

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=data_collator,
            shuffle=shuffle,
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

    def create_embedder_for_train(self, temperature: float = 0.05) -> EmbedderForTrain:
        match self.record_type:
            case RecordType.PAIR:
                model = EmbedderForPairInBatchNegTrain(
                    embedder=self.embedder,
                    temperature=temperature,
                    loss_type=InBatchNegLossType.softmax,
                )
            case RecordType.TRIPLET:
                model = EmbedderForTripletInBatchNegTrain(
                    embedder=self.embedder,
                    temperature=temperature,
                    loss_type=InBatchNegLossType.softmax,
                )
            case RecordType.SCORED_PAIR:
                model = EmbedderForScoredPairTrain(
                    embedder=self.embedder,
                    temperature=temperature,
                )
        return model

    @find_executable_batch_size(starting_batch_size=256)
    def run(
        self,
        # Model
        temperature: float = 0.05,
        training_strategy: TrainingStrategy = FullParametersTraining(),
        # Optimizer
        lr: float | None = None,
        weight_decay: float = 1e-3,
        num_warmup_steps: float | None = None,
        # Data
        batch_size: int = 256,
        max_length: int = 512,
        drop_last: bool = False,
        shuffle: bool = False,
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
        self.accelerator = accelerator
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
            shuffle=shuffle,
            max_length=max_length,
            num_workers=num_workers,
        )
        train_dataloader = accelerator.prepare(train_dataloader)
        validation_dataloader = accelerator.prepare(validation_dataloader) if validation_dataloader is not None else None

        model = self.create_embedder_for_train(temperature=temperature)
        model = training_strategy.apply_model(model)
        model = accelerator.prepare(model)

        # Optimizer & LRScheduler
        lr = lr or suggest_lr(self.embedder)  # type: ignore
        accelerator.print(f'Learning rate: {lr}')
        optimizer = create_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)

        if num_warmup_steps is None:
            lr_scheduler = None
        else:
            if isinstance(train_dataloader.dataset, HFIterableDataset):
                lr_scheduler = None
            else:
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

        if self.model_type is not ModelType.custom:
            save_dir = output_dir / 'model'
            self.save_pretrained(save_dir)
            accelerator.print(f'Saving model to {save_dir}')

        unwrapped_model = cast(EmbedderForTrain, accelerator.unwrap_model(model))
        embedder = unwrapped_model.embedder
        return embedder

    def save_pretrained(self, output_dir: Path | str):
        output_dir = Path(output_dir)
        match self.model_type:
            case ModelType.uniem | ModelType.huggingface | ModelType.text2vec:
                embedder = cast(UniemEmbedder, self.embedder)
                embedder.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
            case ModelType.sentence_transformers:
                from sentence_transformers import SentenceTransformer

                embedder = cast(SentenceTransformer, self.embedder)
                embedder.save(str(output_dir))
            case ModelType.custom:
                raise ValueError('model_type is custom, you should save model by yourself')
