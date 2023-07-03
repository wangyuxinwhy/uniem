import functools
import gc
import json
import logging
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterable, Optional, TypeVar

import torch
import typer
import yaml
from accelerate.utils.memory import should_reduce_batch_size

T = TypeVar('T')
logger = logging.getLogger(__name__)


def load_from_config_file(config_file: str | Path) -> dict[str, Any]:
    config_file = str(config_file)
    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    elif config_file.endswith('.json'):
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f'Unknown config file format: {config_file}, only .yaml, .yml and .json are supported')
    return config


def config_file_callback(ctx: typer.Context, param: typer.CallbackParam, param_value: Any):
    if param_value is None:
        return param_value
    try:
        config = load_from_config_file(param_value)
        ctx.default_map = ctx.default_map or {}
        ctx.default_map.update(config)
    except Exception as e:
        raise typer.BadParameter(str(e), ctx=ctx, param=param) from e
    return param_value


ConfigFile = Annotated[
    Optional[Path],
    typer.Option(..., callback=config_file_callback, is_eager=True, help='Config file path, supports yaml and json'),
]


def create_adamw_optimizer(model: torch.nn.Module, lr: float, weight_decay=1e-3):
    parameters = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm', 'layernorm']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def apply_bitfit(model: torch.nn.Module):
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def split_dataset_dict(dataset_dict: dict[str, T]) -> tuple[T, T | None]:
    if isinstance(dataset_dict, dict):
        train_dataset = dataset_dict['train']
        if 'dev' in dataset_dict:
            validation_dataset = dataset_dict['dev']
        elif 'validation' in dataset_dict:
            validation_dataset = dataset_dict['validation']
        else:
            logger.warning(
                'No validation dataset found in dataset_dict, validation dataset key should be either "dev" or "validation"'
            )
            validation_dataset = None
    else:
        train_dataset = dataset_dict
        validation_dataset = None
    return train_dataset, validation_dataset


def find_executable_batch_size(function: Callable | None = None, starting_batch_size: int = 128):
    if function is None:
        return functools.partial(find_executable_batch_size, starting_batch_size=starting_batch_size)

    batch_size = starting_batch_size

    @wraps(function)
    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        is_manually_passed_batch_size = 'batch_size' in kwargs

        if is_manually_passed_batch_size:
            return function(*args, **kwargs)
        else:
            while True:
                if batch_size == 0:
                    raise RuntimeError('No executable batch size found, reached zero.')
                try:
                    kwargs['batch_size'] = batch_size
                    return function(*args, **kwargs)
                except Exception as e:
                    if should_reduce_batch_size(e):
                        gc.collect()
                        torch.cuda.empty_cache()
                        batch_size //= 2
                        print('Reducing batch size to', batch_size)
                    else:
                        raise

    return decorator


def convert_to_readable_string(number: float) -> str:
    if number >= 1e9:
        return f'{number / 1e9:.1f}B'
    elif number >= 1e6:
        return f'{number / 1e6:.1f}M'
    elif number >= 1e3:
        return f'{number / 1e3:.1f}k'
    else:
        return str(number)
