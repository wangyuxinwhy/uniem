from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeAlias

from datasets import DatasetDict
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

Tokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


class MixedPrecisionType(str, Enum):
    fp16 = 'fp16'
    bf16 = 'bf16'
    no = 'no'


@dataclass
class DatasetDescription:
    name: str
    is_symmetric: bool
    domains: list[str]
    instruction_type: str


@dataclass
class UniemDataset:
    load_fn: Callable[[], DatasetDict]
    description: DatasetDescription
