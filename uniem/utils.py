import functools
import gc
import importlib
import json
import logging
from enum import Enum
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import Annotated, Any, Callable, Generator, Iterable, Optional, Sequence, Type, TypeVar, cast

import torch
import typer
import yaml
from accelerate.utils.memory import should_reduce_batch_size
from transformers import AutoModel, PreTrainedModel

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ConfigFileType(str, Enum):
    yaml = 'yaml'
    json = 'json'


def load_from_yaml(yaml_file: str | Path) -> dict[str, Any]:
    yaml_file = Path(yaml_file)
    if not yaml_file.exists():
        raise FileExistsError(f'File {yaml_file} does not exist')

    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def load_from_json(json_file: str | Path) -> dict[str, Any]:
    json_file = Path(json_file)
    if not json_file.exists():
        raise FileExistsError(f'File {json_file} does not exist')

    with open(json_file, 'r') as f:
        return json.load(f)


def load_config_file(config_file: str | Path, file_type: ConfigFileType | str | None = None) -> dict[str, Any]:
    config_file = Path(config_file)

    if file_type is None:
        file_name = config_file.name
        if file_name.endswith('.yaml') or file_name.endswith('.yml'):
            file_type = ConfigFileType.yaml
        elif file_name.endswith('.json'):
            file_type = ConfigFileType.json
        else:
            raise ValueError(f'Unknown config file format: {config_file}, only .yaml, .yml and .json are supported')
    else:
        file_type = ConfigFileType(file_type)

    match file_type:
        case ConfigFileType.yaml:
            config = load_from_yaml(config_file)
        case ConfigFileType.json:
            config = load_from_json(config_file)
    return config


def _config_file_callback(ctx: typer.Context, param: typer.CallbackParam, param_value: Any):
    if param_value is None:
        return param_value
    try:
        config = load_config_file(param_value)
        ctx.default_map = ctx.default_map or {}
        ctx.default_map.update(config)
    except Exception as e:
        raise typer.BadParameter(str(e), ctx=ctx, param=param) from e
    return param_value


ConfigFile = Annotated[
    Optional[Path],
    typer.Option(..., callback=_config_file_callback, is_eager=True, help='Config file path, supports yaml and json'),
]


def load_hf_pretrained_model(
    model_name_or_path: str, model_class: str | None | Type[PreTrainedModel] | Type[AutoModel] = None
) -> PreTrainedModel:
    if model_class is None:
        model_class = AutoModel
    elif isinstance(model_class, str):
        transformers_module = importlib.import_module('transformers')
        model_class = getattr(transformers_module, model_class)

    model = model_class.from_pretrained(model_name_or_path)  # type: ignore
    model = cast(PreTrainedModel, model)
    return model


def create_adamw_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 1e-3,
    no_decay_keywords: Sequence[str] = ('bias', 'LayerNorm', 'layernorm'),
):
    parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in parameters if not any(nd in n for nd in no_decay_keywords)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in parameters if any(nd in n for nd in no_decay_keywords)],
            'weight_decay': 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer


def create_attention_mask_from_input_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return input_ids != pad_token_id


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch


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


def convert_number_to_readable_string(number: float) -> str:
    if number >= 1e9:
        return f'{number / 1e9:.1f}B'
    elif number >= 1e6:
        return f'{number / 1e6:.1f}M'
    elif number >= 1e3:
        return f'{number / 1e3:.1f}k'
    else:
        return str(number)
