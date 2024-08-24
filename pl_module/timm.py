from torch.utils.data import DataLoader

import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

from module.timm import TimmModel


class LitTimm(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()

        self.config = config
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr

        self.model = TimmModel(config, classes=self.config.num_classes)

        self.init_metrics()

    def training_step(self, batch, _):
        inputs, targets = batch

        stage = "train"
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        acc = self.acc[stage](outputs.argmax(dim=-1), targets)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True,
                sync_dist=True)

        return loss

    def training_epoch_end(self, _):
        stage = "train"
        self.log(f"{stage}_acc_epoch", self.acc[stage].compute(),
                prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, _):
        inputs, targets = batch

        stage = "val"
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        acc = self.acc[stage](outputs.argmax(dim=-1), targets)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True,
                sync_dist=True)

        return loss

    def validation_epoch_end(self, _):
        stage = "val"
        self.log(f"{stage}_acc_epoch", self.acc[stage].compute(),
                prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, _, loader_idx):
        inputs, targets = batch

        stage = "test"
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, targets)
        acc = self.acc[stage][loader_idx](outputs.argmax(dim=-1), targets)

        self.log(f"{stage}_loss_{loader_idx}", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc_{loader_idx}", acc, on_step=False, on_epoch=True,
                sync_dist=True)

        return loss

    def test_epoch_end(self, _):
        stage = "test"
        for loader_idx in range(self.config.num_test_sets):
            self.log(f"{stage}_acc_dataloader_idx{loader_idx}_epoch", self.acc[stage][loader_idx].compute(),
                    prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        if self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
            )
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        elif self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=True
            )

        return optimizer

    def init_metrics(self):
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = nn.ModuleList([
            torchmetrics.Accuracy() for _ in range(self.config.num_test_sets)
        ])

        self.acc = {
            "train": self.train_acc,
            "val": self.val_acc,
            "test": self.test_acc,
        }