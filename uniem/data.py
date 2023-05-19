from dataclasses import dataclass
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import torch
from torch.utils.data import Dataset, RandomSampler
from datasets import Dataset as HfDataset

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
    def __init__(
        self,
        medi_data_file: str | Path,
        batch_size: int = 32,
        pair_or_triplet: str = 'triplet',
        with_prompt: bool = True,
        join_with: str = '\n',
        drop_last: bool = True,
    ):
        medi_data = json.load(fp=Path(medi_data_file).open())
        assert pair_or_triplet in ('pair', 'triplet')

        self._task_records_map: dict[str, list[TripletRecord | PairRecord]] = defaultdict(list)
        for record in medi_data:
            taks_name = record['task_name']
            if with_prompt:
                if pair_or_triplet == 'triplet':
                    record = TripletRecord(
                        text=join_with.join(record['query']),
                        text_pos=join_with.join(record['pos']),
                        text_neg=join_with.join(record['neg']),
                    )
                else:
                    record = PairRecord(
                        text=join_with.join(record['query']),
                        text_pos=join_with.join(record['pos']),
                    )
            else:
                if pair_or_triplet == 'triplet':
                    record = TripletRecord(
                        text=record['query'][1],
                        text_pos=record['pos'][1],
                        text_neg=record['neg'][1],
                    )
                else:
                    record = PairRecord(
                        text=record['query'][1],
                        text_pos=record['pos'][1],
                    )
            self._task_records_map[taks_name].append(record)

        self.batched_records = []
        for _, records in self._task_records_map.items():
            buffer = []

            num_samples = (len(records) // batch_size) * batch_size
            if not drop_last and len(records) % batch_size != 0:
                num_samples += batch_size

            if not num_samples:
                self.batched_records.append(records)
                continue

            for i in RandomSampler(records, num_samples=num_samples):
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


@dataclass
class TaskBatchIndex:
    name: str
    batch_index: list[int]


@dataclass
class M3EHfDatsetWithInfo:
    hf_dataset: HfDataset
    name: str
    instruction: str = ''


# Moka Massive Mixed Embedding Dataset
class M3EDataset(Dataset):
    def __init__(
        self,
        datasets: list[M3EHfDatsetWithInfo],
        batch_size: int = 32,
        with_instruction: bool = True,
        drop_last: bool = True,
    ):
        self.name_dataset_map = {dataset.name: dataset.hf_dataset for dataset in datasets}
        self.task_batch_index_list = []
        for dataset in datasets:
            hf_dataset = dataset.hf_dataset
            dataset_name = dataset.name
            num_samples = (len(hf_dataset) // batch_size) * batch_size
            if not drop_last and len(hf_dataset) % batch_size != 0:
                num_samples += batch_size

            if not num_samples:
                continue

            buffer = []
            for i in RandomSampler(hf_dataset, num_samples=num_samples):
                buffer.append(i)
                if len(buffer) == batch_size:
                    self.task_batch_index_list.append(TaskBatchIndex(name=dataset_name, batch_index=buffer))
                    buffer = []
        if with_instruction:
            self.task_instruction_map = {dataset.name: dataset.instruction for dataset in datasets}
        else:
            self.task_instruction_map = None

    def __getitem__(self, index):
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index
        hf_dataset = self.name_dataset_map[task_name]
        records = [hf_dataset[i] for i in batch_index]
        pair_records = []
        for record in records:
            text = record['text']
            text_pos = record['text_pos']
            if self.task_instruction_map is not None:
                text = self.task_instruction_map[task_name] + text
            pair_records.append(PairRecord(text=text, text_pos=text_pos))
        return pair_records

    def __len__(self):
        return len(self.task_batch_index_list)
