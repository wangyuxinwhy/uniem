from enum import Enum
import logging
import os
import time
from itertools import islice
from typing import Any, Generator, Iterable, Optional, Protocol, TypeVar, cast

import torch
import numpy as np
import openai
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO)

T = TypeVar('T')


class MTEBModel(Protocol):
    def encode(self, sentences: list[str], batch_size: int) -> Any:
        ...


class ModelType(str, Enum):
    m3e_small = 'm3e_small'
    m3e_base = 'm3e_base'
    d_meta_soul = 'd_meta_soul'
    uer = 'uer'
    text2vec = 'text2vec'
    openai = 'openai'
    luotuo = 'luotuo'
    erlangshen = 'erlangshen'
    azure = 'azure'


def load_model(model_type: ModelType) -> MTEBModel:
    match model_type:
        case ModelType.m3e_small:
            return SentenceTransformer('moka-ai/m3e-small')
        case ModelType.m3e_base:
            return SentenceTransformer('moka-ai/m3e-base')
        case ModelType.d_meta_soul:
            return SentenceTransformer('DMetaSoul/sbert-chinese-general-v2')
        case ModelType.uer:
            return SentenceTransformer('uer/sbert-base-chinese-nli')
        case ModelType.text2vec:
            from text2vec import SentenceModel

            return SentenceModel()
        case ModelType.openai:
            return OpenAIModel(model_name='text-embedding-ada-002')
        case ModelType.luotuo:
            return LuotuoBertModel(model_name='silk-road/luotuo-bert')
        case ModelType.erlangshen:
            return ErLangShenModel(model_name='IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
        case _:
            raise ValueError(f'Unknown model type: {model_type}')


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class OpenAIModel:
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'text-embedding-ada-002') -> None:
        if api_key is not None:
            openai.api_key = api_key
        self._client = openai.Embedding
        self.model_name = model_name

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings = []
        for batch in tqdm(generate_batch(sentences, batch_size), total=len(sentences) // batch_size):
            embeddings = self._client.create(input=batch, engine=self.model_name)['data']   # type: ignore
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
        self._client = openai.Embedding
        self.model_name = model_name

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings = []
        for text in tqdm(sentences):
            embeddings = self._client.create(input=text, engine=self.model_name)['data']   # type: ignore
            embeddings = [np.array(result['embedding']) for result in embeddings]
            all_embeddings.extend(embeddings)
            time.sleep(0.01)
        return all_embeddings


class ErLangShenModel:
    def __init__(self, model_name: str = 'IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese', device: str | None = None) -> None:
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for batch_texts in tqdm(generate_batch(sentences, batch_size), total=len(sentences) // batch_size):
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1][:, 0, :].squeeze()
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.extend(embeddings.cpu().numpy())
        return all_embeddings


class LuotuoBertModel:
    def __init__(self, model_name: str = 'silk-road/luotuo-bert', device: str | None = None) -> None:
        from transformers import AutoTokenizer, AutoModel
        from argparse import Namespace

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_args = Namespace(do_mlm=None, pooler_type='cls', temp=0.05, mlp_only_train=False, init_embeddings_model=None)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, model_args=model_args)
        self.model.to(device)

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for batch_texts in tqdm(generate_batch(sentences, batch_size), total=len(sentences) // batch_size):
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = inputs.to(self.device)
            with torch.no_grad():
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.extend(embeddings.cpu().numpy())
        return all_embeddings
