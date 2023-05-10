from enum import Enum
from typing import TypeAlias

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

Tokenizer: TypeAlias = PreTrainedTokenizer | PreTrainedTokenizerFast


class MixedPrecisionType(str, Enum):
    fp16 = 'fp16'
    bf16 = 'bf16'
    no = 'no'
