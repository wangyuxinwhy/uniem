"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

import logging
import os

import typer
from mteb import MTEB
from uniem import Uniem

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TASK_LIST_CLASSIFICATION = [
    'AmazonCounterfactualClassification',
    'AmazonPolarityClassification',
    'AmazonReviewsClassification',
    'Banking77Classification',
    'EmotionClassification',
    'ImdbClassification',
    'MassiveIntentClassification',
    'MassiveScenarioClassification',
    'MTOPDomainClassification',
    'MTOPIntentClassification',
    'ToxicConversationsClassification',
    'TweetSentimentExtractionClassification',
]

TASK_LIST_CLUSTERING = [
    'ArxivClusteringP2P',
    'ArxivClusteringS2S',
    'BiorxivClusteringP2P',
    'BiorxivClusteringS2S',
    'MedrxivClusteringP2P',
    'MedrxivClusteringS2S',
    'RedditClustering',
    'RedditClusteringP2P',
    'StackExchangeClustering',
    'StackExchangeClusteringP2P',
    'TwentyNewsgroupsClustering',
]

TASK_LIST_PAIR_CLASSIFICATION = [
    'SprintDuplicateQuestions',
    'TwitterSemEval2015',
    'TwitterURLCorpus',
]

TASK_LIST_RERANKING = [
    'AskUbuntuDupQuestions',
    'MindSmallReranking',
    'SciDocsRR',
    'StackOverflowDupQuestions',
]

TASK_LIST_RETRIEVAL = [
    'ArguAna',
    'ClimateFEVER',
    'CQADupstackAndroidRetrieval',
    'CQADupstackEnglishRetrieval',
    'CQADupstackGamingRetrieval',
    'CQADupstackGisRetrieval',
    'CQADupstackMathematicaRetrieval',
    'CQADupstackPhysicsRetrieval',
    'CQADupstackProgrammersRetrieval',
    'CQADupstackStatsRetrieval',
    'CQADupstackTexRetrieval',
    'CQADupstackUnixRetrieval',
    'CQADupstackWebmastersRetrieval',
    'CQADupstackWordpressRetrieval',
    'DBPedia',
    'FEVER',
    'FiQA2018',
    'HotpotQA',
    'MSMARCO',
    'NFCorpus',
    'NQ',
    'QuoraRetrieval',
    'SCIDOCS',
    'SciFact',
    'Touche2020',
    'TRECCOVID',
]

TASK_LIST_STS = [
    'BIOSSES',
    'SICK-R',
    'STS12',
    'STS13',
    'STS14',
    'STS15',
    'STS16',
    'STS17',
    'STS22',
    'STSBenchmark',
    'SummEval',
]

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)


def evaluate(uniem_model_name_or_path: str, output_dir: str | None = None):

    model = Uniem.from_pretrained(uniem_model_name_or_path)
    output_dir = output_dir or f'results/{uniem_model_name_or_path.split("/")[-1]}'

    for task in TASK_LIST:
        logger.info(f'Running task: {task}')
        eval_splits = ['dev'] if task == 'MSMARCO' else ['test']
        evaluation = MTEB(tasks=[task], task_langs=['en'])   # Remove "en" for running all languages
        evaluation.run(model, output_folder=output_dir, eval_splits=eval_splits)


if __name__ == '__main__':
    typer.run(evaluate)
