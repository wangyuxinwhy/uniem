from enum import Enum
from pathlib import Path

import torch
from transformers import AutoModel, PreTrainedModel

from uniem.criteria import (
    PairSigmoidContrastLoss,
    PairSoftmaxContrastLoss,
    TripletSigmoidContrastLoss,
    TripletSoftmaxContrastLoss,
)


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


class Embedder(torch.nn.Module):
    def __init__(self, encoder: PreTrainedModel, pad_token_id: int | None = None):
        super().__init__()
        self.encoder = encoder

        if pad_token_id is None:
            if encoder.config.pad_token_id is not None:
                self.pad_token_id = encoder.config.pad_token_id
            else:
                self.pad_token_id = 0
        else:
            self.pad_token_id = pad_token_id

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError

    def save_pretrained(self, path: str | Path):
        self.encoder.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str | Path):
        encoder = AutoModel.from_pretrained(path)
        return cls(encoder)


class LastMeanEmbedder(Embedder):
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids).last_hidden_state
        embeddings = mean_pooling(embeddings, mask)
        return embeddings


class ClsEmbedder(Embedder):
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids).last_hidden_state[:, 0]
        return embeddings


class FirstLastEmbedder(Embedder):
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids, output_hidden_states=True).hidden_states
        first_embeddings = mean_pooling(embeddings[0], mask)
        last_embeddings = mean_pooling(embeddings[-1], mask)
        embeddings = (first_embeddings + last_embeddings) / 2
        return embeddings


class EmbeddingLastEmbedder(Embedder):
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


embedder_map = {
    EmbeddingStrategy.cls: ClsEmbedder,
    EmbeddingStrategy.last_mean: LastMeanEmbedder,
    EmbeddingStrategy.first_last_mean: FirstLastEmbedder,
    EmbeddingStrategy.embedding_last_mean: EmbeddingLastEmbedder,
}


class EmbedderForPairTrain(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.05,
        use_sigmoid: bool = False,
        embedding_strategy: EmbeddingStrategy | str = EmbeddingStrategy.last_mean,
    ):
        super().__init__()
        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
        self.embedding_model = embedder_map[EmbeddingStrategy(embedding_strategy)](pretrained_model)
        if use_sigmoid:
            self.criterion = PairSigmoidContrastLoss(temperature)
        else:
            self.criterion = PairSoftmaxContrastLoss(temperature)

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedding_model(text_ids)
        text_pos_embeddings = self.embedding_model(text_pos_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings)
        return {'loss': loss}


class EmbedderForTripletTrain(torch.nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        temperature: float = 0.05,
        use_sigmoid: bool = False,
        embedding_strategy: EmbeddingStrategy | str = EmbeddingStrategy.last_mean,
    ):
        super().__init__()
        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
        self.embedding_model = embedder_map[EmbeddingStrategy(embedding_strategy)](pretrained_model)
        if use_sigmoid:
            self.criterion = TripletSigmoidContrastLoss(temperature)
        else:
            self.criterion = TripletSoftmaxContrastLoss(temperature)

    def forward(
        self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor, text_neg_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedding_model(text_ids)
        text_pos_embeddings = self.embedding_model(text_pos_ids)
        text_neg_embeddings = self.embedding_model(text_neg_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings)
        return {'loss': loss}
