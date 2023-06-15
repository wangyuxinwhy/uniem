from pathlib import Path

import typer
from mteb import MTEB

from mteb_zh.models import load_model, ModelType
from mteb_zh.tasks import (
    TaskType,
    GubaEastmony,
    IFlyTek,
    JDIphone,
    StockComSentiment,
    T2RRetrieval,
    TNews,
    TYQSentiment,
    T2RReranking,
)


def main(
    model_type: ModelType,
    model_name: str | None = None,
    task_type: TaskType = TaskType.classification,
    output_folder: Path = Path('results'),
):
    output_folder = Path(output_folder)
    model = load_model(model_type, model_name)

    match task_type:
        case TaskType.classification:
            tasks = [
                TYQSentiment(),
                TNews(),
                JDIphone(),
                StockComSentiment(),
                GubaEastmony(),
                IFlyTek(),
            ]
        case TaskType.reranking:
            tasks = [
                T2RReranking(2),
            ]
        case TaskType.retrieval:
            tasks = [
                T2RRetrieval(10000),
            ]
        case TaskType.all:
            tasks = [
                TYQSentiment(),
                TNews(),
                JDIphone(),
                StockComSentiment(),
                GubaEastmony(),
                IFlyTek(),
                T2RReranking(2),
                T2RRetrieval(100000),
            ]
        case _:
            raise ValueError(f'Unknown task type: {task_type}')
    print(tasks)
    evaluation = MTEB(tasks=tasks)
    evaluation.run(model, output_folder=str(output_folder / model_type.value))


if __name__ == '__main__':
    typer.run(main)
