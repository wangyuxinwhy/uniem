import os

import logging

from mteb import MTEB
from mteb.abstasks import AbsTaskClassification
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)

class TNews(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "TNews",
            "hf_hub_name": "clue",
            "description": "clue tnews dataset",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "accuracy",
            "samples_per_label": 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset("clue", "tnews")
        dataset = dataset.rename_column('sentence', 'text')
        self.dataset = dataset
        self.data_loaded = True


class TYQSentiment(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "TYQSentiment",
            "hf_hub_name": "tyqiangz/multilingual-sentiments",
            "description": "A collection of multilingual sentiments datasets grouped into 3 classes -- positive, neutral, negative.",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "accuracy",
            "samples_per_label": 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset('tyqiangz/multilingual-sentiments', 'chinese')
        self.dataset = dataset
        self.data_loaded = True


class IFlyTek(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "IFlyTek",
            "hf_hub_name": "clue",
            "description": "clue iflytek",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "accuracy",
            "samples_per_label": 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset("clue", "iflytek")
        dataset = dataset.rename_column('sentence', 'text')
        self.dataset = dataset
        self.data_loaded = True


class JDIphone(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "JDIphone",
            "hf_hub_name": "kuroneko5943/jd21",
            "description": "kuroneko5943/jd21 iphone",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "accuracy",
            "samples_per_label": 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset("kuroneko5943/jd21", "iPhone")
        dataset = dataset.rename_column('sentence', 'text')
        self.dataset = dataset
        self.data_loaded = True


class StockComSentiment(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "StockComSentiment",
            "hf_hub_name": "kuroneko5943/stock11",
            "description": "kuroneko5943/stock11 communication",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "accuracy",
            "samples_per_label": 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset("kuroneko5943/stock11", "communication")
        dataset = dataset.rename_column('sentence', 'text')
        self.dataset = dataset
        self.data_loaded = True


class GubaEastmony(AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "GubaEastmony",
            "hf_hub_name": "Fearao/guba_eastmoney",
            "description": "数据来自东方财富股吧的评论，经过人工label",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "accuracy",
            "samples_per_label": 32,
        }

    def load_data(self, **kwargs):
        dataset = load_dataset("Fearao/guba_eastmoney")
        self.dataset = dataset
        self.data_loaded = True


def load_m3e_model(m3e_name):
    return SentenceTransformer(f"moka-ai/{m3e_name}")


name = 'm3e-base'

if 'm3e' in name:
    model = load_m3e_model(name)
elif 'text2vec' in name:
    from text2vec import SentenceModel
    model = SentenceModel()
else:
    raise ValueError(f'Unknown model name: {name}')

evaluation = MTEB(tasks=[TYQSentiment(), TNews(), JDIphone(), StockComSentiment(), GubaEastmony(), IFlyTek()])
evaluation.run(model, output_folder=f'results/{name}')
