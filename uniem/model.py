from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Type, TypeVar, cast

import numpy as np
import torch
import tqdm
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel  # type: ignore

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
from uniem.utils import create_attention_mask_from_input_ids, generate_batch, load_hf_pretrained_model

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


def mean_pooling(hidden_state: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if attention_mask is None:
        return torch.mean(hidden_state, dim=1)
    attention_mask = attention_mask.float()
    return torch.sum(hidden_state * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask, dim=-1, keepdim=True)


StrategyEmbedderClsMap: dict[PoolingStrategy, Type['UniemEmbedder']] = {}


class Embedder(Protocol):
    def __call__(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        ...


class UniemEmbedder(torch.nn.Module, Embedder):
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
        config = AutoConfig.from_pretrained(str(model_name_or_path))
        if hasattr(config, 'uniem_pooling_strategy'):
            strategy_string = config.uniem_pooling_strategy
        elif hasattr(config, 'uniem_embedding_strategy'):
            strategy_string = config.uniem_embedding_strategy
        else:
            raise ValueError('Can not find uniem pooling strategy in config, Model is not trained by UniEmbedder.')
        return create_uniem_embedder(str(model_name_or_path), pooling_strategy=strategy_string)

    @property
    def max_length(self):
        return self.encoder.config.max_position_embeddings


class LastMeanEmbedder(UniemEmbedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_mean

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = create_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        embeddings = mean_pooling(embeddings, attention_mask)
        return embeddings


class ClsEmbedder(UniemEmbedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.cls

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = create_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return embeddings


class FirstLastEmbedder(UniemEmbedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.first_last_mean

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = create_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states
        first_embeddings = mean_pooling(embeddings[0], attention_mask)
        last_embeddings = mean_pooling(embeddings[-1], attention_mask)
        embeddings = (first_embeddings + last_embeddings) / 2
        return embeddings


class EmbeddingLastEmbedder(UniemEmbedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.embedding_last_mean

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__(encoder, pad_token_id)
        self.embedding_layer = self.encoder.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = create_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        static_embeddings = self.embedding_layer(input_ids)
        mean_last_embeddings = mean_pooling(
            self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state, attention_mask
        )
        mean_static_embeddings = mean_pooling(static_embeddings, attention_mask)
        return (mean_last_embeddings + mean_static_embeddings) / 2


class LastWeightedEmbedder(UniemEmbedder):
    pooling_strategy: ClassVar[PoolingStrategy] = PoolingStrategy.last_weighted

    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__(encoder, pad_token_id)
        self.embedding_layer = self.encoder.get_input_embeddings()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = create_attention_mask_from_input_ids(input_ids, self.pad_token_id)
        weights = (torch.arange(input_ids.shape[1], device=input_ids.device) + 1).float()
        embeddings = self.encoder(input_ids).last_hidden_state
        embeddings = embeddings * attention_mask.unsqueeze(-1).float() * weights.unsqueeze(0).unsqueeze(-1)
        embeddings = torch.sum(embeddings, dim=1) / torch.sum(weights * attention_mask, dim=-1, keepdim=True)
        return embeddings


def create_uniem_embedder(
    model_name_or_path: str,
    model_class: str | None = None,
    pooling_strategy: PoolingStrategy | str = PoolingStrategy.last_mean,
):
    pretrained_model = load_hf_pretrained_model(model_name_or_path, model_class=model_class)
    embedder_cls = StrategyEmbedderClsMap[PoolingStrategy(pooling_strategy)]
    embedder = embedder_cls(pretrained_model)
    return embedder


class EmbedderForTrain(torch.nn.Module):
    embedder: Embedder

    def __init__(
        self,
        embedder: Embedder,
        criterion: torch.nn.Module,
    ):
        super().__init__()
        self.embedder = embedder
        self.criterion = criterion


class EmbedderForPairInBatchNegTrain(EmbedderForTrain):
    def __init__(
        self,
        embedder: Embedder,
        temperature: float = 0.05,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
    ):
        self.loss_type = InBatchNegLossType(loss_type)
        match self.loss_type:
            case InBatchNegLossType.sigmoid:
                criterion = PairInBatchNegSigmoidContrastLoss(temperature)
            case InBatchNegLossType.softmax:
                criterion = PairInBatchNegSoftmaxContrastLoss(temperature)
            case InBatchNegLossType.cosent:
                criterion = PairInBatchNegCoSentLoss(temperature)
        super().__init__(embedder, criterion)

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings)
        return {'loss': loss}


class EmbedderForTripletInBatchNegTrain(EmbedderForTrain):
    def __init__(
        self,
        embedder: Embedder,
        temperature: float = 0.05,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
        add_swap_loss: bool = False,
    ):
        self.loss_type = InBatchNegLossType(loss_type)
        match self.loss_type:
            case InBatchNegLossType.sigmoid:
                criterion = TripletInBatchNegSigmoidContrastLoss(temperature, add_swap_loss)
            case InBatchNegLossType.softmax:
                criterion = TripletInBatchNegSoftmaxContrastLoss(temperature, add_swap_loss)
            case InBatchNegLossType.cosent:
                criterion = TripletInBatchNegCoSentLoss(temperature, add_swap_loss)
        super().__init__(embedder, criterion)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_pos_ids: torch.Tensor,
        text_neg_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)
        text_neg_embeddings = self.embedder(text_neg_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings)
        return {'loss': loss}


class EmbedderForScoredPairTrain(EmbedderForTrain):
    def __init__(
        self,
        embedder: Embedder,
        temperature: float = 0.05,
    ):
        super().__init__(embedder, CoSentLoss(temperature))

    def forward(
        self,
        text_ids: torch.Tensor,
        text_pair_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pair_ids)
        predict_labels = torch.cosine_similarity(text_embeddings, text_pos_embeddings, dim=-1)
        loss = self.criterion(predict_labels, labels)
        return {'loss': loss, 'predict_labels': predict_labels}


class Uniem:
    PROGRESS_BAR_THRESHOLD = 1000

    def __init__(
        self,
        embedder: UniemEmbedder,
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

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        progress_bar: Literal['auto'] | bool = 'auto',
    ):
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
                batch_embeddings = self.embedder(input_ids, attention_mask=attention_mask)
                if self.normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, dim=-1)
                batch_embeddings = cast(torch.Tensor, batch_embeddings)
            embeddings.extend([i.cpu().numpy() for i in batch_embeddings])
        return embeddings

    def encode_single(self, sentence: str):
        return self.encode([sentence])[0]

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        embedder = UniemEmbedder.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(embedder, tokenizer, **kwargs)

    def save_pretrained(self, ouptut_dir: str):
        self.embedder.save_pretrained(ouptut_dir)
        self.tokenizer.save_pretrained(ouptut_dir)
