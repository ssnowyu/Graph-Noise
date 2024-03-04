from typing import Optional
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from lightning import LightningDataModule
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader

import os
from src.data.components.citeseer import CiteSeerDataset


class CiteseerDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/citeseer",
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        reverse_edge: bool = False,
        noise_type: str = "none",  # ["none", "feat", "missing-edge", "redundant-edge", "error-edge"]
        feat_noise_rate: float = 0.0,
        feat_sigma: float = 0.0,
        edge_rate: float = 0.0,
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

        self.dataset = CiteSeerDataset(
            raw_dir=raw_dir,
            reverse_edge=self.hparams.reverse_edge,
            noise_type=self.hparams.noise_type,
            feat_noise_rate=self.hparams.feat_noise_rate,
            feat_sigma=self.hparams.feat_sigma,
            edge_rate=self.hparams.edge_rate,
            force_reload=True,
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
