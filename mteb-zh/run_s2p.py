import csv
import logging
import sys
from pathlib import Path

import typer
from models import load_model_by_name
from mteb import MTEB
from mteb.abstasks import AbsTaskRetrieval

from datasets import load_dataset

csv.field_size_limit(sys.maxsize)
logging.basicConfig(level=logging.INFO)


def load_t2ranking_for_retraviel():
    collection_dataset = load_dataset('THUIR/T2Ranking', 'collection')['train']  # type: ignore
    dev_queries_dataset = load_dataset('THUIR/T2Ranking', 'queries.dev')['train']  # type: ignore
    dev_rels_dataset = load_dataset('THUIR/T2Ranking', 'qrels.dev')['train']  # type: ignore
    corpus = {}
    for record in collection_dataset:
        corpus[str(record['pid'])] = {'text': record['text']}   # type: ignore
    queries = {}
    for record in dev_queries_dataset:
        queries[str(record['qid'])] = record['text']  # type: ignore
    qrels = {}
    for record in dev_rels_dataset:
        qrels[str(record['qid'])] = {str(record['pid']): record['rel']}  # type: ignore
    return corpus, queries, qrels


class T2RRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'T2RankingRetrieval',
            'reference': 'https://huggingface.co/datasets/THUIR/T2Ranking',
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['dev'],
            'eval_langs': ['zh'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        corpus, queries, qrels = load_t2ranking_for_retraviel()
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        self.corpus['dev'] = corpus
        self.queries['dev'] = queries
        self.relevant_docs['dev'] = qrels
        self.data_loaded = True


def main(name: str, output_folder: Path = Path('s2p')):
    output_folder = Path(output_folder)
    model = load_model_by_name(name)
    evaluation = MTEB(tasks=[T2RRetrieval()])
    evaluation.run(model, output_folder=str(output_folder / name))


if __name__ == '__main__':
    typer.run(main)
