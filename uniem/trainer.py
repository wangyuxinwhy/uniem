from __future__ import annotations

from typing import Any, Sized

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        accelerator: Accelerator,
        validation_dataloader: DataLoader | None = None,
        epochs: int = 3,
        lr_scheduler: LRScheduler | None = None,
        log_interval: int = 50,
        save_on_epoch_end: bool = True,
        epoch_end_callbacks: list[Any] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_on_epoch_end = save_on_epoch_end

        self.train_loss_tracker = LossTracker()
        self.validation_loss_tracker = LossTracker()
        if isinstance(self.train_dataloader.dataset, Sized):
            num_steps_per_epoch = len(self.train_dataloader)
        else:
            num_steps_per_epoch = None
        self.progress_bar = DistributedTqdmProgressBar(self.epochs, num_steps_per_epoch=num_steps_per_epoch)
        self.epoch_end_callbacks = epoch_end_callbacks or []
        self.current_step = 0

    def train(self):
        for current_epoch in range(1, self.epochs + 1):
            self.model.train()
            self.progress_bar.on_epoch_start()

            for batch_index, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    batch_output = self.model(**batch)
                    loss = batch_output['loss']
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.train_loss_tracker.update(loss)

                self.progress_bar.update()
                self.current_step += 1
                if batch_index % self.log_interval == 0:
                    self.log_metrics(
                        {'loss': self.train_loss_tracker.loss},
                        step=self.current_step,
                    )

            train_metrics = self.add_prefix({'loss': self.train_loss_tracker.loss}, 'train')
            self.accelerator.log(train_metrics, step=current_epoch)
            self.train_loss_tracker.on_epoch_end()
            self.progress_bar.on_epoch_end()

            if self.validation_dataloader:
                validation_loss = evaluate(
                    self.model,
                    self.validation_dataloader,
                    self.validation_loss_tracker,
                )
                validation_metrics = self.add_prefix({'loss': validation_loss}, 'validation')
                self.accelerator.print(f'Epoch {current_epoch} Validation loss: {validation_loss:.4f}')
                self.accelerator.log(validation_metrics, step=current_epoch)

            if self.save_on_epoch_end:
                self.accelerator.save_state()

            if self.epoch_end_callbacks:
                for callback in self.epoch_end_callbacks:
                    callback(self)

        self.accelerator.end_training()

    def log_metrics(self, metrics: dict[str, float], step: int):
        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)

    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_tracker: LossTracker | None = None,
):
    model = model.eval()
    loss_tracker = loss_tracker or LossTracker()
    for batch in dataloader:
        with torch.inference_mode():
            batch_output = model(**batch)
            loss_tracker.update(batch_output['loss'])
    loss = loss_tracker.loss
    loss_tracker.on_epoch_end()
    return loss


class DummyProgressBar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def set_description(self, description: str) -> None:
        pass


class DistributedTqdmProgressBar:
    def __init__(self, epochs: int, num_steps_per_epoch: int | None, **kwargs) -> None:
        self.accelerator = Accelerator()
        self.epochs = epochs
        self.current_epoch = 1
        self.num_steps_per_epoch = num_steps_per_epoch
        self.tqdm_kwargs = kwargs

    def on_epoch_start(self):
        if self.accelerator.is_main_process:
            self.progress_bar = tqdm(total=self.num_steps_per_epoch, **self.tqdm_kwargs)
        else:
            self.progress_bar = DummyProgressBar()

    def update(self, n: int = 1) -> None:
        self.progress_bar.update(n)

    def close(self) -> None:
        self.progress_bar.close()

    def on_epoch_end(self) -> None:
        self.current_epoch += 1
        self.progress_bar.close()

    def show_metrics(self, metrics: dict[str, float]) -> None:
        description = f'Epoch {self.current_epoch}/{self.epochs}'
        for name, score in metrics.items():
            description += f' - {name}: {score:.4f}'
        self.progress_bar.set_description(description)


class LossTracker:
    def __init__(
        self,
        ndigits=4,
    ) -> None:
        self.ndigits = ndigits
        self._loss: float = 0.0
        self.loss_count: int = 0
        self.history: list[float] = []

    def update(self, loss_tensor: torch.Tensor):
        loss = loss_tensor.item()
        self._loss = (self._loss * self.loss_count + loss) / (self.loss_count + 1)
        self.loss_count += 1

    def reset(self):
        self._loss = 0
        self.loss_count = 0

    def on_epoch_end(self, reset: bool = True):
        self.history.append(self.loss)
        if reset:
            self.reset()

    @property
    def loss(self) -> float:
        return round(float(self._loss), self.ndigits)
