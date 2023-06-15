from itertools import islice
from typing import Generator, Iterable, TypeVar

import torch

T = TypeVar('T')


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
