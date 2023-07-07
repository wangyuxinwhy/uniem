import json
import os
import time
from enum import Enum
from itertools import islice
from typing import Any, Generator, Iterable, Optional, Protocol, TypeVar, cast

import numpy as np
import openai
import requests
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

T = TypeVar('T')


class MTEBModel(Protocol):
    def encode(self, sentences: list[str], batch_size: int) -> Any:
        ...


class ModelType(str, Enum):
    sentence_transformer = 'sentence_transformer'
    text2vec = 'text2vec'
    luotuo = 'luotuo'
    erlangshen = 'erlangshen'
    openai = 'openai'
    minimax = 'minimax'
    azure = 'azure'


def load_model(model_type: ModelType, model_id: str | None = None) -> MTEBModel:
    match model_type:
        case ModelType.sentence_transformer:
            if model_id is None:
                raise ValueError('model_name must be specified for sentence_transformer')
            return SentenceTransformer(model_id)
        case ModelType.text2vec:
            try:
                from text2vec import SentenceModel  # type: ignore
            except ImportError:
                raise ImportError('text2vec is not installed, please install it with "pip install text2vec"')

            if model_id is None:
                return SentenceModel()
            else:
                return SentenceModel(model_id)
        case ModelType.openai:
            if model_id is None:
                return OpenAIModel(model_name='text-embedding-ada-002')
            else:
                return OpenAIModel(model_name=model_id)
        case ModelType.azure:
            if model_id is None:
                return AzureModel(model_name='text-embedding-ada-002')
            else:
                return AzureModel(model_name=model_id)
        case ModelType.luotuo:
            if model_id is None:
                return LuotuoBertModel(model_name='silk-road/luotuo-bert')
            else:
                return LuotuoBertModel(model_name=model_id)
        case ModelType.erlangshen:
            if model_id is None:
                return ErLangShenModel(model_name='IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
            else:
                return ErLangShenModel(model_name=model_id)
        case ModelType.minimax:
            if model_id is None:
                return MiniMaxModel()
            else:
                if model_id not in {'db', 'query'}:
                    raise ValueError(f'Unknown model type: {model_id}')
                return MiniMaxModel(embedding_type=model_id)
        case _:
            raise ValueError(f'Unknown model type: {model_type}')


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class MiniMaxModel:
    def __init__(self, embedding_type: str = 'db', group_id: str | None = None, api_key: str | None = None) -> None:
        self.embedding_type = embedding_type
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.url = f'https://api.minimax.chat/v1/embeddings?GroupId={self.group_id}'

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}

        embeddings = []
        for batch_sentence in tqdm(generate_batch(sentences, batch_size), total=len(sentences) // batch_size):
            data = {'texts': batch_sentence, 'model': 'embo-01', 'type': 'db'}
            response = requests.post(self.url, headers=headers, data=json.dumps(data)).json()
            for embedding in response['vectors']:
                embeddings.append(np.array(embedding))
        return embeddings


class OpenAIModel:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'text-embedding-ada-002',
    ) -> None:
        if api_key is not None:
            openai.api_key = api_key
        self._client = openai.Embedding.create
        self.model_name = model_name

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings = []
        for batch in tqdm(
            generate_batch(sentences, batch_size),
            total=len(sentences) // batch_size,
        ):
            output = self._client(input=batch, engine=self.model_name)
            embeddings = output['data']   # type: ignore
            embeddings = sorted(embeddings, key=lambda e: e['index'])  # type: ignore
            embeddings = [np.array(result['embedding']) for result in embeddings]
            all_embeddings.extend(embeddings)
        return all_embeddings


class AzureModel:
    def __init__(self, model_name: str = 'text-embedding-ada-002') -> None:
        openai.api_type = 'azure'
        openai.api_key = os.environ['AZURE_API_KEY']
        openai.api_base = os.environ['AZURE_API_BASE']
        openai.api_version = '2023-03-15-preview'
        self._client = openai.Embedding.create
        self.model_name = model_name

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings = []
        for text in tqdm(sentences):
            output = self._client(input=text, engine=self.model_name)
            embeddings = output['data']  # type: ignore
            embeddings = [np.array(result['embedding']) for result in embeddings]
            all_embeddings.extend(embeddings)
            time.sleep(0.01)
        return all_embeddings


class ErLangShenModel:
    def __init__(
        self,
        model_name: str = 'IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese',
        device: str | None = None,
    ) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for batch_texts in tqdm(
            generate_batch(sentences, batch_size),
            total=len(sentences) // batch_size,
        ):
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1][:, 0, :].squeeze()
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.extend(embeddings.cpu().numpy())
        return all_embeddings


class LuotuoBertModel:
    def __init__(
        self,
        model_name: str = 'silk-road/luotuo-bert',
        device: str | None = None,
    ) -> None:
        from argparse import Namespace

        from transformers import AutoModel, AutoTokenizer  # type: ignore

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_args = Namespace(
            do_mlm=None,
            pooler_type='cls',
            temp=0.05,
            mlp_only_train=False,
            init_embeddings_model=None,
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, model_args=model_args)
        self.model.to(device)

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for batch_texts in tqdm(
            generate_batch(sentences, batch_size),
            total=len(sentences) // batch_size,
        ):
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            )
            inputs = inputs.to(self.device)
            with torch.no_grad():
                embeddings = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True,
                ).pooler_output
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.extend(embeddings.cpu().numpy())
        return all_embeddings
