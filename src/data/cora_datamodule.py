from typing import Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from lightning import LightningDataModule
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader

import os
from src.data.components.cora import CoraDataset


class CoraDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/cora",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        reverse_edge: bool = True,
        noise_rate: float = 0.0,
        sigma: float = 0.01,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes() -> int:
        return 7

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        raw_dir = os.path.join(self.hparams.data_dir, "raw")
        # save_dir = os.path.join(self.hparams.data_dir, "processed")

        self.dataset = CoraDataset(
            raw_dir=raw_dir,
            reverse_edge=self.hparams.reverse_edge,
            noise_rate=self.hparams.noise_rate,
            sigma=self.hparams.sigma,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return GraphDataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return GraphDataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return GraphDataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
