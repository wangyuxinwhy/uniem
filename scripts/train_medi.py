import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Literal, Optional, cast

import typer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader, Dataset, RandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from uniem.data import TripletCollator
from uniem.data_structures import TripletRecord
from uniem.model import UniEmbeddingModelForTripletTrain
from uniem.trainer import Trainer
from uniem.types import MixedPrecisionType
from uniem.utils import create_adamw_optimizer


class MediDataset(Dataset):
    def __init__(self, data_file: str | Path, batch_size: int):
        medi_data = json.load(fp=Path(data_file).open())

        self._task_records_map = defaultdict(list)
        for record in medi_data:
            taks_name = record['task_name']
            record = TripletRecord(
                text='\n'.join(record['query']),
                text_pos='\n'.join(record['pos']),
                text_neg='\n'.join(record['neg']),
            )
            self._task_records_map[taks_name].append(record)

        self.batched_records = []
        for _, v in self._task_records_map.items():
            buffer = []
            for i in RandomSampler(v, num_samples=(1 + len(v) // batch_size) * batch_size):
                buffer.append(v[i])
                if len(buffer) == batch_size:
                    self.batched_records.append(buffer)
                    buffer = []

        self.batch_size = batch_size

    def __getitem__(self, index):
        return self.batched_records[index]

    def __len__(self):
        return len(self.batched_records)


def main(
    model_name_or_path: str,
    medi_data_file: Path,
    # Model
    temperature: Annotated[float, typer.Option(rich_help_panel='Model')] = 0.05,
    use_sigmoid: Annotated[bool, typer.Option(rich_help_panel='Model')] = False,
    max_length: Annotated[int, typer.Option(rich_help_panel='Model')] = 512,
    # Optimizer
    lr: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 3e-5,
    weight_decay: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 1e-3,
    num_warmup_steps: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 0.05,
    # Trainer
    epochs: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 3,
    batch_size: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 32,
    mixed_precision: Annotated[MixedPrecisionType, typer.Option(rich_help_panel='Trainer')] = MixedPrecisionType.no,
    gradient_accumulation_steps: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 1,
    save_on_epoch_end: Annotated[bool, typer.Option(rich_help_panel='Trainer')] = False,
    num_max_checkpoints: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 1,
    use_tensorboard: Annotated[bool, typer.Option(rich_help_panel='Trainer')] = False,
    num_workers: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 4,
    seed: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 42,
    output_dir: Annotated[Optional[Path], typer.Option(rich_help_panel='Trainer')] = None,
):
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    if num_workers > 1:
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    output_dir = output_dir or Path('experiments') / 'medi'
    project_config = ProjectConfiguration(
        project_dir=str(output_dir), automatic_checkpoint_naming=True, total_limit=num_max_checkpoints
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision.value,
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_config=project_config,
        log_with=['tensorboard'] if use_tensorboard else None,
    )
    accelerator.init_trackers(f'medi')

    set_seed(seed)
    accelerator.print(f'Start with seed: {seed}')
    accelerator.print(f'Output dir: {output_dir}')

    # DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = MediDataset(data_file=medi_data_file, batch_size=batch_size)
    data_collator = TripletCollator(tokenizer=tokenizer, max_length=max_length)
    train_dataloader = DataLoader(
        train_dataset, batch_size=None, collate_fn=data_collator, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    # Model
    model = UniEmbeddingModelForTripletTrain(
        model_name_or_path=model_name_or_path, temperature=temperature, use_sigmoid=use_sigmoid
    )
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
        validation_dataloader=None,
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
    unwrapped_model = cast(UniEmbeddingModelForTripletTrain, accelerator.unwrap_model(model))

    unwrapped_model.embedding_model.save_pretrained(output_dir / 'model')
    tokenizer.save_pretrained(output_dir / 'model')


if __name__ == '__main__':
    typer.run(main)
