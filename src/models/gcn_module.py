from typing import Any, Dict, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
import dgl
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)


class GCNModule(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        num_classes: int,
        optimizer: torch.optim.Optimizer,
        compile: bool,
        scheduler=None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["encoder"])

        self.encoder = encoder

        self.criterion = torch.nn.CrossEntropyLoss()

        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_precision = MulticlassPrecision(num_classes=num_classes)
        self.val_recall = MulticlassRecall(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes)

        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_precision = MulticlassPrecision(num_classes=num_classes)
        self.test_recall = MulticlassRecall(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, g: dgl.DGLGraph, feat):
        h = self.encoder(g, feat)
        return h

    def training_step(self, batch: dgl.DGLGraph, batch_idx: int) -> STEP_OUTPUT:
        g: dgl.DGLGraph = batch
        feat = g.ndata["feat"]
        labels = g.ndata["label"]
        train_mask = g.ndata["train_mask"]

        logits = self.forward(g, feat)
        logits = self.sigmoid(logits)

        loss = self.criterion(logits[train_mask], labels[train_mask])

        self.train_loss(loss)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
        )

        return loss

    def validation_step(self, batch, idx) -> Optional[STEP_OUTPUT]:
        g: dgl.DGLGraph = batch
        features = g.ndata["feat"]
        labels = g.ndata["label"]
        val_mask = g.ndata["val_mask"]

        logits = self.forward(g, features)

        preds = logits[val_mask]
        # preds = preds.argmax(1)
        targets = labels[val_mask]

        self.val_acc(preds, targets)
        self.val_precision(preds, targets)
        self.val_recall(preds, targets)
        self.val_f1(preds, targets)

        self.log("val/acc", self.val_acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val/precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/recall", self.val_recall, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log("val/f1", self.val_f1, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, idx) -> Optional[STEP_OUTPUT]:
        g: dgl.DGLGraph = batch
        feat = g.ndata["feat"]
        labels = g.ndata["label"]
        test_mask = g.ndata["test_mask"]

        logits = self.forward(g, feat)

        preds = logits[test_mask]
        # preds = preds.argmax(1)
        targets = labels[test_mask]

        self.test_acc(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.test_f1(preds, targets)

        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/precision",
            self.test_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "preds": preds,
            "targets": targets,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
