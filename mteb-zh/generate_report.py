import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer


task_score_field_mapping: dict[str, tuple[str, str]] = {
    'T2Reranking': ('dev', 'map'),
    'GubaEastmony': ('test', 'accuracy'),
    'IFlyTek': ('validation', 'accuracy'),
    'JDIphone': ('validation', 'accuracy'),
    'StockComSentiment': ('validation', 'accuracy'),
    'TNews': ('validation', 'accuracy'),
    'TYQSentiment': ('validation', 'accuracy'),
    'T2RankingRetrieval': ('dev', 'ndcg_at_10'),
}


def generate_report_csv(results_dir: Path, output_file: Path = Path('m3e-evaluate.csv')):
    scores = defaultdict(list)

    for dir in results_dir.iterdir():
        model_name = dir.name
        for path in dir.glob('*.json'):
            data = json.load(path.open())
            name = data['mteb_dataset_name']
            field = task_score_field_mapping[name]
            score: float = data[field[0]][field[1]]
            scores[name].append((model_name, round(score, 4)))

    df = pd.DataFrame()
    for dataset, model_scores in scores.items():
        for model, score in model_scores:
            df.loc[dataset, model] = score
    df.loc['Average'] = df.mean(axis=0)
    df.to_csv(output_file)


if __name__ == '__main__':
    typer.run(generate_report_csv)
