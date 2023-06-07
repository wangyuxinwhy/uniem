import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import typer


def generate_report_csv(results_dir: Path, output_file: Path = Path('m3e-evaluate.csv')):
    scores = defaultdict(list)

    for dir in results_dir.iterdir():
        model_name = dir.name
        for path in dir.glob('*.json'):
            data = json.load(path.open())
            name = data['mteb_dataset_name']
            if 'test' in data:
                score: float = data['test']['accuracy']
            else:
                score: float = data['validation']['accuracy']
            scores[name].append((model_name, round(score, 4)))

    df = pd.DataFrame()
    for dataset, model_scores in scores.items():
        for model, score in model_scores:
            df.loc[dataset, model] = score
    df.loc['Average'] = df.mean(axis=0)
    df.to_csv(output_file)


if __name__ == '__main__':
    typer.run(generate_report_csv)
