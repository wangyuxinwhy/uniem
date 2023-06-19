from pathlib import Path
from typing import Annotated

import typer
from mteb import MTEB, AbsTask
from mteb_zh.models import ModelType, load_model
from mteb_zh.tasks import (
    GubaEastmony,
    IFlyTek,
    JDIphone,
    MedQQPairs,
    StockComSentiment,
    T2RReranking,
    T2RRetrieval,
    TaskType,
    TNews,
    TYQSentiment,
)

default_tasks: list[AbsTask] = [
    TYQSentiment(),
    TNews(),
    JDIphone(),
    StockComSentiment(),
    GubaEastmony(),
    IFlyTek(),
    T2RReranking(2),
    T2RRetrieval(10000),
    MedQQPairs(),
]


def main(
    model_type: Annotated[ModelType, typer.Option()],
    model_id: str | None = None,
    model_name: str | None = None,
    task_type: TaskType = TaskType.Classification,
    output_folder: Path = Path('results'),
):
    output_folder = Path(output_folder)
    model = load_model(model_type, model_id)

    if task_type is TaskType.All:
        tasks = default_tasks
    else:
        tasks = [task for task in default_tasks if task.description['type'] == task_type.value]  # type: ignore

    evaluation = MTEB(tasks=tasks)
    if model_name is None:
        model_name = model_type.value + (f'-{model_id.replace("/", "-")}' if model_id else '')
    evaluation.run(model, output_folder=str(output_folder / model_name))


if __name__ == '__main__':
    typer.run(main)
