import os
from pathlib import Path
from typing import Annotated, Optional, cast

import typer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from uniem.data import MediDataset, TripletCollator
from uniem.model import EmbedderForTripletTrain, EmbeddingStrategy
from uniem.trainer import Trainer
from uniem.types import MixedPrecisionType
from uniem.utils import create_adamw_optimizer

app = typer.Typer()


@app.command()
def main(
    model_name_or_path: str,
    medi_data_file: Path,
    # Model
    temperature: Annotated[float, typer.Option(rich_help_panel='Model')] = 0.05,
    use_sigmoid: Annotated[bool, typer.Option(rich_help_panel='Model')] = False,
    max_length: Annotated[int, typer.Option(rich_help_panel='Model')] = 512,
    embedding_strategy: Annotated[EmbeddingStrategy, typer.Option(rich_help_panel='Model')] = EmbeddingStrategy.last_mean,
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
    num_workers: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 0,
    seed: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 42,
    output_dir: Annotated[Optional[Path], typer.Option(rich_help_panel='Trainer')] = None,
):
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')
    if num_workers >= 1:
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
    accelerator.init_trackers('medi')

    set_seed(seed)
    accelerator.print(f'Start with seed: {seed}')
    accelerator.print(f'Output dir: {output_dir}')

    # DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = MediDataset(medi_data_file=medi_data_file, batch_size=batch_size)
    data_collator = TripletCollator(tokenizer=tokenizer, max_length=max_length)
    train_dataloader = DataLoader(
        train_dataset, batch_size=None, collate_fn=data_collator, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    # Model
    model = EmbedderForTripletTrain(
        model_name_or_path=model_name_or_path,
        temperature=temperature,
        use_sigmoid=use_sigmoid,
        embedding_strategy=embedding_strategy,
    )
    model.embedding_model.encoder.config.pad_token_id = tokenizer.pad_token_id
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
    unwrapped_model = cast(EmbedderForTripletTrain, accelerator.unwrap_model(model))

    unwrapped_model.embedding_model.save_pretrained(output_dir / 'model')
    tokenizer.save_pretrained(output_dir / 'model')


if __name__ == '__main__':
    app()
