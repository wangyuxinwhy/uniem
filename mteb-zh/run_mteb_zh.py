from pathlib import Path
from typing import Annotated

import typer
from mteb import MTEB, AbsTask

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


default_tasks: list[AbsTask] = [
    TYQSentiment(),
    TNews(),
    JDIphone(),
    StockComSentiment(),
    GubaEastmony(),
    IFlyTek(),
    T2RReranking(2),
    T2RRetrieval(10000),
]


def main(
    model_type: Annotated[ModelType, typer.Option()],
    model_name: str | None = None,
    task_type: TaskType = TaskType.Classification,
    output_folder: Path = Path('results'),
):
    output_folder = Path(output_folder)
    model = load_model(model_type, model_name)

    if task_type is TaskType.All:
        tasks = default_tasks
    else:
        tasks = [task for task in default_tasks if task.description()['type'] == task_type.value]

    evaluation = MTEB(tasks=tasks)
    model_id = model_type.value + (f'-{model_name.replace("/", "-")}' if model_name else '')
    evaluation.run(model, output_folder=str(output_folder / model_id))


if __name__ == '__main__':
    typer.run(main)
