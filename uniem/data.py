import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import Dataset, RandomSampler

from uniem.data_structures import PairRecord, TripletRecord
from uniem.types import Tokenizer


class PairCollator:
    def __init__(self, tokenizer: Tokenizer, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, records: list[PairRecord]) -> dict[str, torch.Tensor]:
        texts = [record.text for record in records]
        texts_pos = [record.text_pos for record in records]

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_ids = cast(torch.Tensor, text_ids)

        text_pos_ids = self.tokenizer(
            texts_pos,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pos_ids = cast(torch.Tensor, text_pos_ids)

        return {
            'text_ids': text_ids,
            'text_pos_ids': text_pos_ids,
        }


class TripletCollator:
    def __init__(self, tokenizer: Tokenizer, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, records: list[TripletRecord]) -> dict[str, torch.Tensor]:
        texts = [record.text for record in records]
        texts_pos = [record.text_pos for record in records]
        texts_neg = [record.text_neg for record in records]

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pos_ids = self.tokenizer(
            texts_pos,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_neg_ids = self.tokenizer(
            texts_neg,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']

        text_ids = cast(torch.Tensor, text_ids)
        text_pos_ids = cast(torch.Tensor, text_pos_ids)
        text_neg_ids = cast(torch.Tensor, text_neg_ids)
        return {
            'text_ids': text_ids,
            'text_pos_ids': text_pos_ids,
            'text_neg_ids': text_neg_ids,
        }


class MediDataset(Dataset):
    def __init__(self, medi_data_file: str | Path, batch_size: int, join_with: str = '\n'):
        medi_data = json.load(fp=Path(medi_data_file).open())

        self._task_records_map: dict[str, list[TripletRecord]] = defaultdict(list)
        for record in medi_data:
            taks_name = record['task_name']
            record = TripletRecord(
                text=join_with.join(record['query']),
                text_pos=join_with.join(record['pos']),
                text_neg=join_with.join(record['neg']),
            )
            self._task_records_map[taks_name].append(record)

        self.batched_records = []
        for _, records in self._task_records_map.items():
            buffer = []
            for i in RandomSampler(records, num_samples=(1 + len(records) // batch_size) * batch_size):
                buffer.append(records[i])
                if len(buffer) == batch_size:
                    self.batched_records.append(buffer)
                    buffer = []

        self.batch_size = batch_size
        self.join_with = join_with

    def __getitem__(self, index):
        return self.batched_records[index]

    def __len__(self):
        return len(self.batched_records)
