from enum import Enum
from itertools import islice
from pathlib import Path
from typing import ClassVar, Generator, Iterable, Literal, Type, TypeVar, cast

import torch
import numpy as np
import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel

from uniem.criteria import (
    PairSigmoidContrastLoss,
    PairSoftmaxContrastLoss,
    TripletSigmoidContrastLoss,
    TripletSoftmaxContrastLoss,
)
from uniem.types import Tokenizer

T = TypeVar('T')


class EmbeddingStrategy(str, Enum):
    cls = 'cls'
    last_mean = 'last_mean'
    first_last_mean = 'first_last_mean'
    embedding_last_mean = 'embedding_last_mean'


def creat_mask_from_input_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return input_ids != pad_token_id


def mean_pooling(hidden_state: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return torch.mean(hidden_state, dim=1)
    mask = mask.float()
    return torch.sum(hidden_state * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=-1, keepdim=True)


StrategyEmbedderClsMap: dict[EmbeddingStrategy, Type['Embedder']] = {}


class Embedder(torch.nn.Module):
    embedding_strategy: ClassVar[EmbeddingStrategy]

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.encoder.config.uniem_embedding_strategy = str(self.embedding_strategy.value)

        if pad_token_id is None:
            if encoder.config.pad_token_id is not None:
                self.pad_token_id = encoder.config.pad_token_id
            else:
                self.pad_token_id = 0
        else:
            self.pad_token_id = pad_token_id

    def __init_subclass__(cls) -> None:
        StrategyEmbedderClsMap[cls.embedding_strategy] = cls

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def save_pretrained(self, path: str | Path):
        self.encoder.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str | Path):
        encoder = AutoModel.from_pretrained(path)
        return cls(encoder)

    @property
    def max_length(self):
        return self.encoder.config.max_position_embeddings


class LastMeanEmbedder(Embedder):
    embedding_strategy: ClassVar[EmbeddingStrategy] = EmbeddingStrategy.last_mean

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids).last_hidden_state
        embeddings = mean_pooling(embeddings, mask)
        return embeddings


class ClsEmbedder(Embedder):
    embedding_strategy: ClassVar[EmbeddingStrategy] = EmbeddingStrategy.cls

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids).last_hidden_state[:, 0]
        return embeddings


class FirstLastEmbedder(Embedder):
    embedding_strategy: ClassVar[EmbeddingStrategy] = EmbeddingStrategy.first_last_mean

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, output_hidden_states=True).hidden_states
        first_embeddings = mean_pooling(embeddings[0], mask)
        last_embeddings = mean_pooling(embeddings[-1], mask)
        embeddings = (first_embeddings + last_embeddings) / 2
        return embeddings


class EmbeddingLastEmbedder(Embedder):
    embedding_strategy: ClassVar[EmbeddingStrategy] = EmbeddingStrategy.embedding_last_mean

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__(encoder, pad_token_id)
        self.embedding_layer = self.encoder.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        static_embeddings = self.embedding_layer(input_ids)
        mean_last_embeddings = mean_pooling(self.encoder(input_ids).last_hidden_state, mask)
        mean_static_embeddings = mean_pooling(static_embeddings, mask)
        return (mean_last_embeddings + mean_static_embeddings) / 2


class AutoEmbedder:
    @classmethod
    def from_pretrained(cls, path: str | Path):
        config = AutoConfig.from_pretrained(path)
        if config.model_type == 't5':
            from transformers import T5EncoderModel

            encoder = T5EncoderModel.from_pretrained(path)
        else:
            encoder = AutoModel.from_pretrained(path)
        encoder = cast(PreTrainedModel, encoder)
        embedder_cls = StrategyEmbedderClsMap[EmbeddingStrategy(encoder.config.uniem_embedding_strategy)]
        return embedder_cls(encoder)


class EmbedderForTrain(torch.nn.Module):
    def __init__(self, embedder: Embedder, chunk_size: int = 8):
        super().__init__()
        self.embedder = embedder
        self.chunk_size = chunk_size

    def chunk_embedder_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        num_chunks = (input_ids.size(0) // self.chunk_size) + 1
        if num_chunks <= 1:
            return self.embedder(input_ids)

        chunks = torch.chunk(input_ids, num_chunks, dim=0)
        embeddings = [self.embedder(chunk) for chunk in chunks]
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings


class EmbedderForPairTrain(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.05,
        use_sigmoid: bool = False,
        embedding_strategy: EmbeddingStrategy | str = EmbeddingStrategy.last_mean,
        chunk_size: int = 8,
    ):
        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
        embedder = StrategyEmbedderClsMap[EmbeddingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder, chunk_size)
        if use_sigmoid:
            self.criterion = PairSigmoidContrastLoss(temperature)
        else:
            self.criterion = PairSoftmaxContrastLoss(temperature)

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings)
        return {'loss': loss}


class EmbedderForTripletTrain(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.05,
        use_sigmoid: bool = False,
        embedding_strategy: EmbeddingStrategy | str = EmbeddingStrategy.last_mean,
        chunk_size: int = 8,
    ):
        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
        embedder = StrategyEmbedderClsMap[EmbeddingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder, chunk_size)
        if use_sigmoid:
            self.criterion = TripletSigmoidContrastLoss(temperature)
        else:
            self.criterion = TripletSoftmaxContrastLoss(temperature)
        self.chunk_size = chunk_size

    def forward(
        self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor, text_neg_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.chunk_embedder_forward(text_ids)
        text_pos_embeddings = self.chunk_embedder_forward(text_pos_ids)
        text_neg_embeddings = self.chunk_embedder_forward(text_neg_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings)
        return {'loss': loss}


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class UniEmbedder:
    PROGRESS_BAR_THRESHOLD = 1000

    def __init__(self, embedder: Embedder, tokenizer: Tokenizer, max_legnth: int | None = None, device: str | None = None):
        super().__init__()
        self.embedder = embedder.eval()
        if device:
            self.embedder = self.embedder.to(device)
        self.tokenizer = tokenizer
        self.max_length = max_legnth or self.tokenizer.model_max_length

    def __call__(self, sentences: list[str], batch_size: int = 32):
        return self.encode(sentences, batch_size)

    def encode(self, sentences: list[str], batch_size: int = 32, progress_bar: Literal['auto'] | bool = 'auto'):
        embeddings: list[np.ndarray] = []
        if progress_bar == 'auto':
            progress_bar = len(sentences) > self.PROGRESS_BAR_THRESHOLD

        for batch in tqdm.tqdm(
            generate_batch(sentences, batch_size),
            disable=not progress_bar,
            total=len(sentences) // batch_size,
            unit='batch',
            desc='Encoding',
        ):
            input_ids = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')['input_ids']
            input_ids = cast(torch.Tensor, input_ids)
            input_ids = input_ids.to(self.embedder.encoder.device)
            with torch.inference_mode():
                batch_embeddings = self.embedder(input_ids)
                batch_embeddings = cast(torch.Tensor, batch_embeddings)
            embeddings.extend([i.cpu().numpy() for i in batch_embeddings])
        return embeddings

    def encode_single(self, sentence: str):
        return self.encode([sentence])[0]

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, max_legnth: int | None = None, device: str | None = None):
        encoder = AutoEmbedder.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return cls(encoder, tokenizer, max_legnth=max_legnth, device=device)

    def save_pretrained(self, ouptut_dir: str):
        self.embedder.save_pretrained(ouptut_dir)
        self.tokenizer.save_pretrained(ouptut_dir)
