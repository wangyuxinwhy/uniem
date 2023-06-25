import os
from pathlib import Path
from typing import Annotated, Optional, cast

import typer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import Dataset as HfDataset
from datasets import concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup  # type: ignore
from uniem.data import M3EDataset, M3EHfDatsetWithInfo, PairCollator
from uniem.model import (
    EmbedderForPairInBatchNegTrain,
    EmbedderForTrain,
    InBatchNegLossType,
    PoolingStrategy,
)
from uniem.trainer import Trainer
from uniem.types import MixedPrecisionType
from uniem.utils import convert_to_readable_string, create_adamw_optimizer

app = typer.Typer()


def load_all_datasets(m3e_datasets_dir: Path) -> list[M3EHfDatsetWithInfo]:
    m3e_datasets = []
    for data_dir in m3e_datasets_dir.glob('*.dataset'):
        dataset_name = data_dir.stem
        dataset_dict = load_from_disk(str(data_dir))
        if isinstance(dataset_dict, dict):
            dataset: HfDataset = concatenate_datasets(list(dataset_dict.values()))
        else:
            dataset = dataset_dict
        m3e_datasets.append(
            M3EHfDatsetWithInfo(
                hf_dataset=dataset,
                name=dataset_name,
            )
        )
        print(f'load {dataset_name}')
    return m3e_datasets


@app.command()
def main(
    model_name_or_path: str,
    m3e_datasets_dir: Path,
    # Model
    model_class: Annotated[Optional[str], typer.Option(rich_help_panel='Model')] = None,
    temperature: Annotated[float, typer.Option(rich_help_panel='Model')] = 0.05,
    loss_type: Annotated[InBatchNegLossType, typer.Option(rich_help_panel='Model')] = InBatchNegLossType.softmax,
    embedding_strategy: Annotated[PoolingStrategy, typer.Option(rich_help_panel='Model')] = PoolingStrategy.last_mean,
    # Data
    batch_size: Annotated[int, typer.Option(rich_help_panel='Data')] = 32,
    with_instruction: Annotated[bool, typer.Option(rich_help_panel='Data')] = True,
    drop_last: Annotated[bool, typer.Option(rich_help_panel='Data')] = True,
    max_length: Annotated[int, typer.Option(rich_help_panel='Data')] = 512,
    # Optimizer
    lr: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 3e-5,
    weight_decay: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 1e-3,
    num_warmup_steps: Annotated[float, typer.Option(rich_help_panel='Optimizer')] = 0.05,
    # Trainer
    epochs: Annotated[int, typer.Option(rich_help_panel='Trainer')] = 3,
    bitfit: Annotated[bool, typer.Option(rich_help_panel='Trainer')] = False,
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

    output_dir = output_dir or Path('experiments') / 'm3e'
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
    accelerator.init_trackers('m3e')

    set_seed(seed)
    accelerator.print(f'Start with seed: {seed}')
    accelerator.print(f'Output dir: {output_dir}')

    # DataLoader
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    all_m3e_datasets = load_all_datasets(m3e_datasets_dir)
    train_dataset = M3EDataset(
        all_m3e_datasets,
        batch_size=batch_size,
        with_instruction=with_instruction,
        drop_last=drop_last,
    )
    data_collator = PairCollator(tokenizer=tokenizer, max_length=max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    model = EmbedderForPairInBatchNegTrain(
        model_name_or_path=model_name_or_path,
        model_class=model_class,
        temperature=temperature,
        loss_type=loss_type,
        embedding_strategy=embedding_strategy,
    )
    if bitfit:
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    num_training_paramters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f'Number of training parameters: {convert_to_readable_string(num_training_paramters)}')
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

    def refresh_data(trainer: Trainer):
        train_dataset.create_or_refresh_data()

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
        epoch_end_callbacks=[refresh_data],
    )
    accelerator.print(f'Start training for {epochs} epochs')
    trainer.train()

    accelerator.wait_for_everyone()
    accelerator.print('Training finished')

    accelerator.print('Saving model')
    unwrapped_model = cast(EmbedderForTrain, accelerator.unwrap_model(model))

    unwrapped_model.embedder.save_pretrained(output_dir / 'model')
    tokenizer.save_pretrained(output_dir / 'model')


if __name__ == '__main__':
    app()
