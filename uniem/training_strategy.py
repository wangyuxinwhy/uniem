import functools
from typing import Sequence, TypeVar, cast

import torch

from uniem.data import FinetuneDataset, FinetuneIterableDataset, PrefixFinetuneDataset, PrefixFinetuneIterableDataset
from uniem.model import EmbedderForTrain, UniemEmbedder
from uniem.types import Tokenizer

T = TypeVar('T', bound=EmbedderForTrain)


class TrainingStrategy:
    def apply_model(self, model: T) -> T:
        raise NotImplementedError


class FullParametersTraining(TrainingStrategy):
    def apply_model(self, model: T) -> T:
        return model


class BitFitTrainging(TrainingStrategy):
    def __init__(self, keywords: Sequence[str] | str = 'bias') -> None:
        if isinstance(keywords, str):
            self.keywords = [keywords]
        else:
            self.keywords = list(keywords)

    def apply_model(self, model: T) -> T:
        for name, param in model.named_parameters():
            if any(word in name for word in self.keywords):
                param.requires_grad = True
            else:
                param.requires_grad = False
        return model


def partial_freeze_gradients(grad, train_indices: torch.Tensor):
    train_indices_grad = grad[train_indices, :]
    grad.zero_()
    grad[train_indices, :] = train_indices_grad
    return grad


class PrefixTraining(TrainingStrategy):
    tokenizer: Tokenizer
    additional_special_token_ids: list[int]

    def __init__(
        self,
        additional_special_tokens: list[str],
        prefix: str | None = None,
        only_train_additional_special_tokens: bool = True,
    ) -> None:
        self.additional_special_tokens = additional_special_tokens
        self.prefix = ''.join(self.additional_special_tokens) if prefix is None else prefix
        self.only_train_additional_special_tokens = only_train_additional_special_tokens

    def apply_tokenizer(self, tokenizer: Tokenizer):
        tokenizer.add_special_tokens({'additional_special_tokens': self.additional_special_tokens})  # type: ignore
        additional_special_token_ids = tokenizer.convert_tokens_to_ids(self.additional_special_tokens)
        if isinstance(additional_special_token_ids, int):
            additional_special_token_ids = [additional_special_token_ids]
        self.additional_special_token_ids = additional_special_token_ids
        self.tokenizer = tokenizer
        return tokenizer

    def apply_model(self, model: T) -> T:
        embedder = model.embedder
        if not isinstance(embedder, UniemEmbedder):
            raise ValueError('Prefix training is only supported for UniemEmbedder')
        embedder = cast(UniemEmbedder, embedder)

        embedder.encoder.resize_token_embeddings(len(self.tokenizer))
        hook = functools.partial(
            partial_freeze_gradients,
            train_indices=torch.tensor(self.additional_special_token_ids),
        )
        if self.only_train_additional_special_tokens:
            for param in model.parameters():
                param.requires_grad = False
            embedding_layer_weight = embedder.encoder.get_input_embeddings().weight
            embedding_layer_weight = cast(torch.nn.Parameter, embedding_layer_weight)
            embedding_layer_weight.requires_grad = True
            embedding_layer_weight.register_hook(hook)
        return model

    def apply_dataset(self, dataset: FinetuneDataset | FinetuneIterableDataset):
        if isinstance(dataset, FinetuneDataset):
            return PrefixFinetuneDataset(dataset=dataset.dataset, prefix=self.prefix, record_type=dataset.record_type)
        else:
            return PrefixFinetuneIterableDataset(dataset=dataset.dataset, prefix=self.prefix, record_type=dataset.record_type)
