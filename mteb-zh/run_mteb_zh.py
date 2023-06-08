from pathlib import Path

import typer
from models import load_model_by_name
from mteb import MTEB

from tasks import GubaEastmony, IFlyTek, JDIphone, StockComSentiment, T2RRetrieval, TNews, TYQSentiment, T2RReranking


def main(name: str, output_folder: Path = Path('results')):
    output_folder = Path(output_folder)
    model = load_model_by_name(name)
    evaluation = MTEB(
        tasks=[TYQSentiment(), TNews(), JDIphone(), StockComSentiment(), GubaEastmony(), IFlyTek(), T2RReranking(), T2RRetrieval()]
    )
    evaluation.run(model, output_folder=str(output_folder / name))

if __name__ == '__main__':
    typer.run(main)