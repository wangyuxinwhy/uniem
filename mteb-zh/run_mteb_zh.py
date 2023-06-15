from pathlib import Path

import typer
from models import load_model, ModelType
from mteb import MTEB

from tasks import GubaEastmony, IFlyTek, JDIphone, StockComSentiment, T2RRetrieval, TNews, TYQSentiment, T2RReranking


def main(model_type: ModelType, model_name: str | None = None, output_folder: Path = Path('results')):
    output_folder = Path(output_folder)
    model = load_model(model_type)
    evaluation = MTEB(
        tasks=[
            TYQSentiment(),
            TNews(),
            JDIphone(),
            StockComSentiment(),
            GubaEastmony(),
            IFlyTek(),
            T2RReranking(2),
            T2RRetrieval(100000),
        ]
    )
    evaluation.run(model, output_folder=str(output_folder / model_type.value))


if __name__ == '__main__':
    typer.run(main)
