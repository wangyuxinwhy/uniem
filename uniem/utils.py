import logging
from itertools import islice
from typing import Generator, Iterable, TypeVar

import torch

T = TypeVar('T')
logger = logging.getLogger(__name__)


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
