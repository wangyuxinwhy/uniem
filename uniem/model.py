from pathlib import Path

import torch
from transformers import AutoModel, PreTrainedModel

from uniem.criteria import (
    PairSigmoidContrastLoss,
    PairSoftmaxContrastLoss,
    TripletSigmoidContrastLoss,
    TripletSoftmaxContrastLoss,
)


def creat_mask_from_input_ids(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return input_ids != pad_token_id


class MeanPooler(torch.nn.Module):
    def forward(self, last_hidden_states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return torch.mean(last_hidden_states, dim=1)
        mask = mask.float()
        return torch.sum(last_hidden_states * mask.unsqueeze(-1), dim=1) / torch.sum(mask, dim=-1, keepdim=True)


class UniEmbeddingModel(torch.nn.Module):
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

        self.pooler = MeanPooler()

    def forward(self, input_ids: torch.Tensor):
        mask = creat_mask_from_input_ids(input_ids, self.pad_token_id)
        embeddings = self.encoder(input_ids).last_hidden_state
        embeddings = self.pooler(embeddings, mask)
        return embeddings

    def save_pretrained(self, path: str | Path):
        self.encoder.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str | Path):
        encoder = AutoModel.from_pretrained(path)
        return cls(encoder)


class UniEmbeddingModelForPairTrain(torch.nn.Module):
    def __init__(self, model_name_or_path: str, temperature: float = 0.05, use_sigmoid: bool = False):
        super().__init__()
        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
        self.embedding_model = UniEmbeddingModel(pretrained_model)
        if use_sigmoid:
            self.criterion = PairSigmoidContrastLoss(temperature)
        else:
            self.criterion = PairSoftmaxContrastLoss(temperature)

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedding_model(text_ids)
        text_pos_embeddings = self.embedding_model(text_pos_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings)
        return {'loss': loss}


class UniEmbeddingModelForTripletTrain(torch.nn.Module):
    def __init__(self, model_name_or_path: str, temperature: float = 0.05, use_sigmoid: bool = False):
        super().__init__()
        pretrained_model = AutoModel.from_pretrained(model_name_or_path)
        self.embedding_model = UniEmbeddingModel(pretrained_model)
        if use_sigmoid:
            self.criterion = TripletSigmoidContrastLoss(temperature)
        else:
            self.criterion = TripletSoftmaxContrastLoss(temperature)

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor, text_neg_ids: torch.Tensor)  -> dict[str, torch.Tensor]:
        text_embeddings = self.embedding_model(text_ids)
        text_pos_embeddings = self.embedding_model(text_pos_ids)
        text_neg_embeddings = self.embedding_model(text_neg_ids)
        loss = self.criterion(text_embeddings, text_pos_embeddings, text_neg_embeddings)
        return {'loss': loss}
