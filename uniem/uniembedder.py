from typing import Literal, cast

import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer

from uniem.model import AutoEmbedder, Embedder
from uniem.types import Tokenizer
from uniem.utils import generate_batch


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
