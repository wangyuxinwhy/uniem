import logging
import os
import time
from itertools import islice
from typing import Generator, Iterable, Optional, TypeVar, cast

import torch
import numpy as np
import openai
from tqdm import tqdm

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO)

T = TypeVar('T')


def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
    iterator = iter(data)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def load_model_by_name(name: str):
    if 'm3e' in name:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(f'moka-ai/{name}')
    elif 'text2vec' in name:
        from text2vec import SentenceModel

        model = SentenceModel()
    elif 'openai' in name:
        from models import OpenAIModel

        model = OpenAIModel()
    elif 'azure' in name:
        from models import AzureModel

        model = AzureModel()
    elif 'luotuo' in name:
        from models import LuotuoBertModel
        if name == 'luotuo':
            model = LuotuoBertModel()
        else:
            model = LuotuoBertModel(name)
    else:
        raise ValueError(f'Unknown model name: {name}')
    return model


class OpenAIModel:
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'text-embedding-ada-002') -> None:
        if api_key is not None:
            openai.api_key = api_key
        self._client = openai.Embedding
        self.model_name = model_name

    def encode(self, texts: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings = []
        for batch in tqdm(generate_batch(texts, batch_size), total=len(texts) // batch_size):
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

    def encode(self, texts: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings = []
        for text in tqdm(texts):
            embeddings = self._client.create(input=text, engine=self.model_name)['data']   # type: ignore
            embeddings = [np.array(result['embedding']) for result in embeddings]
            all_embeddings.extend(embeddings)
            time.sleep(0.01)
        return all_embeddings


class LuotuoBertModel:
    def __init__(self, model_name: str = 'silk-road/luotuo-bert') -> None:
        from transformers import AutoTokenizer, AutoModel
        from argparse import Namespace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False, init_embeddings_model=None)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, model_args=model_args)
    
    def encode(self, texts: list[str], batch_size: int = 32, **kwargs) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for batch_texts in tqdm(self.generate_batch(texts, batch_size), total=len(texts) // batch_size):
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True, sent_emb=True).pooler_output
            embeddings = cast(torch.Tensor, embeddings)
            all_embeddings.extend(embeddings.cpu().numpy())
        return all_embeddings

    @staticmethod
    def generate_batch(data: Iterable[T], batch_size: int = 32) -> Generator[list[T], None, None]:
        iterator = iter(data)
        while batch := list(islice(iterator, batch_size)):
            yield batch
