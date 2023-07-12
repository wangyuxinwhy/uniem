import csv
import math
import sys
from collections import defaultdict
from enum import Enum
from typing import Iterable, TypeVar, cast

from datasets import Dataset, DatasetDict, load_dataset
from mteb.abstasks import (
    AbsTaskClassification,
    AbsTaskPairClassification,
    AbsTaskReranking,
    AbsTaskRetrieval,
)
from tqdm import tqdm

T = TypeVar('T')
csv.field_size_limit(sys.maxsize)


class TaskType(str, Enum):
    Classification = 'Classification'
    Reranking = 'Reranking'
    Retrieval = 'Retrieval'
    PairClassification = 'PairClassification'
    All = 'All'


class MedQQPairs(AbsTaskPairClassification):
    @property
    def description(self):
        return {
            'name': 'MedQQPairs',
            'hf_hub_name': 'vegaviazhang/Med_QQpairs',
            'category': 's2s',
            'type': 'PairClassification',
            'eval_splits': ['train'],
            'eval_langs': ['zh'],
            'main_score': 'ap',
        }

    def load_data(self, **kwargs):
        dataset = load_dataset('vegaviazhang/Med_QQpairs')['train']  # type: ignore
        record = {'sent1': [], 'sent2': [], 'labels': []}
        for item in dataset:
            item = cast(dict, item)
            record['sent1'].append(item['question1'])
            record['sent2'].append(item['question2'])
            record['labels'].append(item['label'])
        self.dataset = DatasetDict(train=Dataset.from_list([record]))
        self.data_loaded = True


class TNews(AbsTaskClassification):
    @property
    def description(self):
        return {
            'name': 'TNews',
            'hf_hub_name': 'clue',
            'description': 'clue tnews dataset',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['validation'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
            'n_experiments': 5,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset('clue', 'tnews')
        dataset = dataset.rename_column('sentence', 'text')
        self.dataset = dataset
        self.data_loaded = True


class TYQSentiment(AbsTaskClassification):
    @property
    def description(self):
        return {
            'name': 'TYQSentiment',
            'hf_hub_name': 'tyqiangz/multilingual-sentiments',
            'description': 'multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative.',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['validation'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset('tyqiangz/multilingual-sentiments', 'chinese')
        self.dataset = dataset
        self.data_loaded = True


class IFlyTek(AbsTaskClassification):
    @property
    def description(self):
        return {
            'name': 'IFlyTek',
            'hf_hub_name': 'clue',
            'description': 'clue iflytek',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['validation'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
            'n_experiments': 3,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset('clue', 'iflytek')
        dataset = dataset.rename_column('sentence', 'text')
        self.dataset = dataset
        self.data_loaded = True


class JDIphone(AbsTaskClassification):
    @property
    def description(self):
        return {
            'name': 'JDIphone',
            'hf_hub_name': 'kuroneko5943/jd21',
            'description': 'kuroneko5943/jd21 iphone',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['validation'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset('kuroneko5943/jd21', 'iPhone')
        dataset = dataset.rename_column('sentence', 'text')
        self.dataset = dataset
        self.data_loaded = True


class StockComSentiment(AbsTaskClassification):
    @property
    def description(self):
        return {
            'name': 'StockComSentiment',
            'hf_hub_name': 'kuroneko5943/stock11',
            'description': 'kuroneko5943/stock11 communication',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['validation'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset('kuroneko5943/stock11', 'communication')
        dataset = dataset.rename_column('sentence', 'text')
        self.dataset = dataset
        self.data_loaded = True


class GubaEastmony(AbsTaskClassification):
    @property
    def description(self):
        return {
            'name': 'GubaEastmony',
            'hf_hub_name': 'Fearao/guba_eastmoney',
            'description': '数据来自东方财富股吧的评论，经过人工label',
            'category': 's2s',
            'type': 'Classification',
            'eval_splits': ['test'],
            'eval_langs': ['zh'],
            'main_score': 'accuracy',
            'samples_per_label': 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset('Fearao/guba_eastmoney')
        self.dataset = dataset
        self.data_loaded = True


def load_t2ranking_for_reranking(rel_threshold: int):
    assert rel_threshold >= 1

    collection_dataset = load_dataset('THUIR/T2Ranking', 'collection')['train']  # type: ignore
    dev_queries_dataset = load_dataset('THUIR/T2Ranking', 'queries.dev')['train']  # type: ignore
    dev_rels_dataset = load_dataset('THUIR/T2Ranking', 'qrels.dev')['train']  # type: ignore
    dev_rels_dataset = cast(Iterable[dict], dev_rels_dataset)
    records = defaultdict(lambda: [[], []])
    query_map = {record['qid']: record['text'] for record in dev_queries_dataset}  # type: ignore

    for rel_record in tqdm(dev_rels_dataset):
        rel_record = cast(dict, rel_record)
        qid = rel_record['qid']
        pid = rel_record['pid']
        rel_score = rel_record['rel']
        query_text = query_map[qid]
        passage_record = collection_dataset[pid]
        assert passage_record['pid'] == pid
        if rel_score >= rel_threshold:
            records[query_text][0].append(passage_record['text'])
        else:
            records[query_text][1].append(passage_record['text'])

    data = [{'query': k, 'positive': v[0], 'negative': v[1]} for k, v in records.items()]
    dataset = Dataset.from_list(data)
    dataset_dict = DatasetDict(dev=dataset)
    return dataset_dict


class T2RReranking(AbsTaskReranking):
    def __init__(self, rel_threshold: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.rel_threshold = rel_threshold

    @property
    def description(self):
        return {
            'name': 'T2RReranking',
            'reference': 'https://huggingface.co/datasets/THUIR/T2Ranking',
            'type': 'Reranking',
            'category': 's2s',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'map',
        }

    def load_data(self, **kwargs):
        dataset = load_t2ranking_for_reranking(self.rel_threshold)
        self.dataset = dataset
        self.data_loaded = True


def load_t2ranking_for_retraviel(num_max_passages: float):
    collection_dataset = load_dataset('THUIR/T2Ranking', 'collection')['train']  # type: ignore
    dev_queries_dataset = load_dataset('THUIR/T2Ranking', 'queries.dev')['train']  # type: ignore
    dev_rels_dataset = load_dataset('THUIR/T2Ranking', 'qrels.dev')['train']  # type: ignore

    corpus = {}
    for record in collection_dataset:
        record = cast(dict, record)
        pid: int = record['pid']
        if pid > num_max_passages:
            break
        corpus[str(pid)] = {'text': record['text']}

    queries = {}
    for record in dev_queries_dataset:
        record = cast(dict, record)
        queries[str(record['qid'])] = record['text']

    all_qrels = defaultdict(dict)
    for record in dev_rels_dataset:
        record = cast(dict, record)
        pid: int = record['pid']
        if pid > num_max_passages:
            continue
        all_qrels[str(record['qid'])][str(record['pid'])] = record['rel']
    valid_qrels = {}
    for qid, qrels in all_qrels.items():
        if len(set(list(qrels.values())) - set([0])) >= 1:
            valid_qrels[qid] = qrels
    valid_queries = {}
    for qid, query in queries.items():
        if qid in valid_qrels:
            valid_queries[qid] = query
    print(f'valid qrels: {len(valid_qrels)}')
    return corpus, valid_queries, valid_qrels


class T2RRetrieval(AbsTaskRetrieval):
    def __init__(self, num_max_passages: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.num_max_passages = num_max_passages or math.inf

    @property
    def description(self):
        return {
            'name': 'T2RRetrieval',
            'reference': 'https://huggingface.co/datasets/THUIR/T2Ranking',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        corpus, queries, qrels = load_t2ranking_for_retraviel(self.num_max_passages)
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        self.corpus['dev'] = corpus
        self.queries['dev'] = queries
        self.relevant_docs['dev'] = qrels
        self.data_loaded = True
