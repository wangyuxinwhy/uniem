import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import torch
from datasets import Dataset as HfDataset
from datasets import IterableDataset as HfIterableDataset
from torch.utils.data import Dataset, IterableDataset, RandomSampler

from uniem.data_structures import (
    PairRecord,
    RecordType,
    ScoredPairRecord,
    TripletRecord,
    infer_record_type,
    record_type_cls_map,
)
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


class ScoredPairCollator:
    def __init__(self, tokenizer: Tokenizer, max_length: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length

    def __call__(self, records: list[ScoredPairRecord]) -> dict[str, torch.Tensor]:
        texts = [record.sentence1 for record in records]
        texts_pair = [record.sentence2 for record in records]
        labels = [record.label for record in records]
        labels = torch.tensor(labels, dtype=torch.float32)

        text_ids = self.tokenizer(
            texts,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_ids = cast(torch.Tensor, text_ids)

        text_pair_ids = self.tokenizer(
            texts_pair,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )['input_ids']
        text_pair_ids = cast(torch.Tensor, text_pair_ids)

        return {
            'text_ids': text_ids,
            'text_pair_ids': text_pair_ids,
            'labels': labels,
        }


class FinetuneDataset(Dataset):
    def __init__(
        self,
        dataset: HfDataset | Sequence[dict],
        record_type: RecordType | str | None = None,
    ) -> None:
        self.dataset = dataset
        if record_type:
            self.record_type = RecordType(record_type)
        else:
            self.record_type = infer_record_type(dataset[0])
        self.record_cls = record_type_cls_map[self.record_type]

    def __getitem__(self, index: int):
        record = self.dataset[index]
        return self.record_cls(**record)

    def __len__(self):
        return len(self.dataset)


class FinetuneIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: HfIterableDataset | Iterable[dict],
        record_type: RecordType | str | None = None,
    ) -> None:
        self.dataset = dataset
        if record_type:
            self.record_type = RecordType(record_type)
        else:
            self.record_type = infer_record_type(next(iter(dataset)))
        self.record_cls = record_type_cls_map[self.record_type]

    def __iter__(self):
        for record in self.dataset:
            yield self.record_cls(**record)


class PrefixFinetuneDataset(FinetuneDataset):
    def __init__(
        self,
        dataset: HfDataset | Sequence[dict],
        prefix: str,
        record_type: RecordType | str | None = None,
    ) -> None:
        super().__init__(dataset=dataset, record_type=record_type)
        self.prefix = prefix

    def __getitem__(self, index: int):
        record = self.dataset[index]
        match self.record_type:
            case RecordType.PAIR:
                record['text'] = self.prefix + record['text']
            case RecordType.TRIPLET:
                record['text'] = self.prefix + record['text']
            case RecordType.SCORED_PAIR:
                record['sentence1'] = self.prefix + record['sentence1']
        return self.record_cls(**record)


class PrefixFinetuneIterableDataset(FinetuneIterableDataset):
    def __init__(
        self,
        dataset: HfIterableDataset | Iterable[dict],
        prefix: str,
        record_type: RecordType | str | None = None,
    ) -> None:
        super().__init__(dataset=dataset, record_type=record_type)
        self.prefix = prefix

    def __iter__(self):
        for record in self.dataset:
            match self.record_type:
                case RecordType.PAIR:
                    record['text'] = self.prefix + record['text']
                case RecordType.TRIPLET:
                    record['text'] = self.prefix + record['text']
                case RecordType.SCORED_PAIR:
                    record['sentence1'] = self.prefix + record['sentence1']
            yield self.record_cls(**record)


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
        self.batch_size = batch_size
        self.join_with = join_with
        self.drop_last = drop_last
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
        self.create_or_refresh_data()

    def create_or_refresh_data(self):
        batch_size = self.batch_size
        self.batched_records = []
        for _, records in self._task_records_map.items():
            buffer = []

            num_samples = (len(records) // batch_size) * batch_size
            if not self.drop_last and len(records) % batch_size != 0:
                num_samples += batch_size

            if not num_samples:
                self.batched_records.append(records)
                continue

            for i in RandomSampler(records, num_samples=num_samples):
                buffer.append(records[i])
                if len(buffer) == batch_size:
                    self.batched_records.append(buffer)
                    buffer = []
        self.random_index_list = list(RandomSampler(self.batched_records))

    def __getitem__(self, index):
        index = self.random_index_list[index]
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
        m3e_hf_datasets: list[M3EHfDatsetWithInfo],
        batch_size: int = 32,
        with_instruction: bool = True,
        drop_last: bool = True,
        max_samples: int | None = None,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.m3e_hf_datasets = m3e_hf_datasets
        self.max_samples = max_samples
        self.name_dataset_map = {dataset.name: dataset.hf_dataset for dataset in m3e_hf_datasets}
        if with_instruction:
            self.task_instruction_map = {dataset.name: dataset.instruction for dataset in m3e_hf_datasets}
        else:
            self.task_instruction_map = None
        self.create_or_refresh_data()

    @staticmethod
    def is_valid_text(text: Any) -> bool:
        return isinstance(text, str) and bool(text.strip())

    def create_or_refresh_data(self):
        self.task_batch_index_list: list[TaskBatchIndex] = []
        for dataset in self.m3e_hf_datasets:
            max_samples = self.max_samples or len(dataset.hf_dataset)
            num_samples = (max_samples // self.batch_size) * self.batch_size
            buffer = []
            for i in RandomSampler(dataset.hf_dataset, num_samples=num_samples):
                buffer.append(i)
                if len(buffer) == self.batch_size:
                    self.task_batch_index_list.append(TaskBatchIndex(name=dataset.name, batch_index=buffer))
                    buffer = []
        self.random_index_list = list(RandomSampler(self.task_batch_index_list))

    def __getitem__(self, index: int):
        index = self.random_index_list[index]
        task_batch_index = self.task_batch_index_list[index]
        task_name = task_batch_index.name
        batch_index = task_batch_index.batch_index
        hf_dataset = self.name_dataset_map[task_name]
        records = [hf_dataset[i] for i in batch_index]
        pair_records = []
        for record in records:
            text = record['text']
            text_pos = record['text_pos']
            if not (self.is_valid_text(text) and self.is_valid_text(text_pos)):
                continue
            if self.task_instruction_map is not None:
                text = self.task_instruction_map[task_name] + text
            pair_records.append(PairRecord(text=text, text_pos=text_pos))
        if not pair_records:
            raise ValueError(f'records is empty, {records}')
        return pair_records

    def __len__(self):
        return len(self.task_batch_index_list)
