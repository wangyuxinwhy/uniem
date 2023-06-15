from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal, Type, TypeVar, cast

import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

from uniem.criteria import (
    CoSentLoss,
    PairInBatchNegCoSentLoss,
    PairInBatchNegSigmoidContrastLoss,
    PairInBatchNegSoftmaxContrastLoss,
    TripletInBatchNegCoSentLoss,
    TripletInBatchNegSigmoidContrastLoss,
    TripletInBatchNegSoftmaxContrastLoss,
)
from uniem.types import Tokenizer
from uniem.utils import generate_batch

T = TypeVar('T')


class PoolingStrategy(str, Enum):
    cls = 'cls'
    last_mean = 'last_mean'
    first_last_mean = 'first_last_mean'
    embedding_last_mean = 'embedding_last_mean'
    last_weighted = 'last_weighted'


class InBatchNegLossType(str, Enum):
    sigmoid = 'sigmoid'
    softmax = 'softmax'
    cosent = 'cosent'


def creat_mask_from_input_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return input_ids != pad_token_id


def mean_pooling(hidden_state: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return torch.mean(hidden_state, dim=1)
    mask = mask.float()
    return torch.sum(hidden_state * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=-1, keepdim=True)


def load_hf_pretrained_model(model_name_or_path: str) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(model_name_or_path)
    if config.model_type == 't5':
        from transformers import T5EncoderModel

        pretrained_model = T5EncoderModel.from_pretrained(model_name_or_path)
    else:
        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
    return pretrained_model  # type: ignore


StrategyEmbedderClsMap: dict[PoolingStrategy, Type['Embedder']] = {}


class Embedder(torch.nn.Module):
    pooling_strategy: ClassVar[PoolingStrategy]

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.encoder.config.uniem_pooling_strategy = str(self.pooling_strategy.value)

        if pad_token_id is None:
            if encoder.config.pad_token_id is not None:
                self.pad_token_id = encoder.config.pad_token_id
            else:
                self.pad_token_id = 0
        else:
            self.pad_token_id = pad_token_id

    def __init_subclass__(cls) -> None:
        StrategyEmbedderClsMap[cls.pooling_strategy] = cls

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def save_pretrained(self, path: str | Path):
        self.encoder.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        encoder = load_hf_pretrained_model(model_name_or_path)
        return cls(encoder)

    @property
    def max_length(self):
        return self.encoder.config.max_position_embeddings


class LastMeanEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_mean

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids).last_hidden_state
        embeddings = mean_pooling(embeddings, mask)
        return embeddings


class ClsEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.cls

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.encoder(input_ids).last_hidden_state[:, 0]
        return embeddings


class FirstLastEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.first_last_mean

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, output_hidden_states=True).hidden_states
        first_embeddings = mean_pooling(embeddings[0], mask)
        last_embeddings = mean_pooling(embeddings[-1], mask)
        embeddings = (first_embeddings + last_embeddings) / 2
        return embeddings


class EmbeddingLastEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.embedding_last_mean

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


class LastWeightedEmbedder(Embedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_weighted

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__(encoder, pad_token_id)
        self.embedding_layer = self.encoder.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        weights = (torch.arange(input_ids.shape[1], device=input_ids.device) + 1).float()
        embeddings = self.encoder(input_ids).last_hidden_state
        embeddings = embeddings * mask.unsqueeze(-1).float() * weights.unsqueeze(0).unsqueeze(-1)
        embeddings = torch.sum(embeddings, dim=1) / torch.sum(weights * mask, dim=-1, keepdim=True)
        return embeddings


class AutoEmbedder:
    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path):
        encoder = load_hf_pretrained_model(str(model_name_or_path))
        if hasattr(encoder.config, 'uniem_pooling_strategy'):
            strategy_string = encoder.config.uniem_pooling_strategy
        elif hasattr(encoder.config, 'uniem_embedding_strategy'):
            strategy_string = encoder.config.uniem_embedding_strategy
        else:
            raise ValueError('Can not find uniem pooling strategy in config, Model is not trained by UniEmbedder.')
        embedder_cls = StrategyEmbedderClsMap[PoolingStrategy(strategy_string)]
        return embedder_cls(encoder)


class EmbedderForTrain(torch.nn.Module):
    def __init__(self, embedder: Embedder):
        super().__init__()
        self.embedder = embedder


class EmbedderForPairInBatchNegTrain(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float | None = None,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder)
        temperature = temperature or 0.05
        self.loss_type = InBatchNegLossType(loss_type)
        match self.loss_type:
            case InBatchNegLossType.sigmoid:
                self.criterion = PairInBatchNegSigmoidContrastLoss(temperature)
            case InBatchNegLossType.softmax:
                self.criterion = PairInBatchNegSoftmaxContrastLoss(temperature)
            case InBatchNegLossType.cosent:
                self.criterion = PairInBatchNegCoSentLoss(temperature)

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings)
        return {'loss': loss}


class EmbedderForTripletInBatchNegTrain(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float | None = None,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
        add_swap_loss: bool = False,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder)
        temperature = temperature or 0.05
        self.loss_type = InBatchNegLossType(loss_type)
        match self.loss_type:
            case InBatchNegLossType.sigmoid:
                self.criterion = TripletInBatchNegSigmoidContrastLoss(temperature, add_swap_loss)
            case InBatchNegLossType.softmax:
                self.criterion = TripletInBatchNegSoftmaxContrastLoss(temperature, add_swap_loss)
            case InBatchNegLossType.cosent:
                self.criterion = TripletInBatchNegCoSentLoss(temperature, add_swap_loss)

    def forward(
        self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor, text_neg_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        text_neg_embeddings = self.embedder(text_neg_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings)
        return {'loss': loss}


class EmbedderForScoredPairTrain(EmbedderForTrain):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float | None = None,
        embedding_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
    ):
        pretrained_model = load_hf_pretrained_model(model_name_or_path)
        embedder = StrategyEmbedderClsMap[PoolingStrategy(embedding_strategy)](pretrained_model)
        super().__init__(embedder)
        temperature = temperature or 0.05
        self.criterion = CoSentLoss(temperature)

    def forward(self, text_ids: torch.Tensor, text_pair_ids: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pair_ids)
        predict_labels = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        loss = self.criterion(predict_labels, labels)
        return {'loss': loss, 'predict_labels': predict_labels}


class UniEmbedder:
    PROGRESS_BAR_THRESHOLD = 1000

    def __init__(
        self,
        embedder: Embedder,
        tokenizer: Tokenizer,
        normalize: bool = True,
        max_length: int | None = None,
        device: str | None = None,
    ):
        super().__init__()
        self.embedder = embedder.eval()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedder = self.embedder.to(device)
        self.tokenizer = tokenizer
        self.normalize = normalize
        self.max_length = (
            max_length or self.embedder.encoder.config.max_length or self.embedder.encoder.config.max_position_embeddings
        )

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
            encodes = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True,
                max_length=self.max_length,
            )

            input_ids = encodes['input_ids']
            input_ids = cast(torch.Tensor, input_ids)
            input_ids = input_ids.to(self.embedder.encoder.device)

            attention_mask = encodes['attention_mask']
            attention_mask = cast(torch.Tensor, attention_mask)
            attention_mask = attention_mask.to(self.embedder.encoder.device)

            with torch.inference_mode():
                batch_embeddings = self.embedder(input_ids, mask=attention_mask)
                if self.normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
                batch_embeddings = cast(torch.Tensor, batch_embeddings)
            embeddings.extend([i.cpu().numpy() for i in batch_embeddings])
        return embeddings

    def encode_single(self, sentence: str):
        return self.encode([sentence])[0]

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        encoder = AutoEmbedder.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(encoder, tokenizer, **kwargs)

    def save_pretrained(self, ouptut_dir: str):
        self.embedder.save_pretrained(ouptut_dir)
        self.tokenizer.save_pretrained(ouptut_dir)
