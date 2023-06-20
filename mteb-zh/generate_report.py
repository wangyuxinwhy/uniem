import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import typer
from mteb_zh.tasks import TaskType


@dataclass
class ReportMainScore:
    split: str
    metric_name: str


task_mapping: dict[TaskType, dict[str, ReportMainScore]] = {
    TaskType.Classification: {
        'GubaEastmony': ReportMainScore('test', 'accuracy'),
        'IFlyTek': ReportMainScore('validation', 'accuracy'),
        'JDIphone': ReportMainScore('validation', 'accuracy'),
        'StockComSentiment': ReportMainScore('validation', 'accuracy'),
        'TNews': ReportMainScore('validation', 'accuracy'),
        'TYQSentiment': ReportMainScore('validation', 'accuracy'),
    },
    TaskType.Reranking: {
        'T2RReranking': ReportMainScore('dev', 'map'),
    },
    TaskType.Retrieval: {
        'T2RankingRetrieval': ReportMainScore('dev', 'ndcg_at_10'),
    },
}
task_mapping[TaskType.All] = {k: v for _, mapping in task_mapping.items() for k, v in mapping.items()}


def generate_report_csv(results_dir: Path, task_type: TaskType = TaskType.Classification):
    scores = defaultdict(list)
    output_file: Path = Path(f'mteb-zh-{task_type.value}.csv')

    mapping = task_mapping[task_type]

    for dir in results_dir.iterdir():
        model_name = dir.name
        for path in dir.glob('*.json'):
            data = json.load(path.open())
            name = data['mteb_dataset_name']
            if name not in mapping:
                continue

            report_main_score = mapping[name]
            score: float = data[report_main_score.split][report_main_score.metric_name]
            scores[name].append((model_name, round(score, 4)))

    df = pd.DataFrame()
    for dataset, model_scores in scores.items():
        for model, score in model_scores:
            df.loc[dataset, model] = score
    df.loc['Average'] = df.mean(axis=0)
    df.to_csv(output_file)


if __name__ == '__main__':
    typer.run(generate_report_csv)
