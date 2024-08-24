import torch
import torchmetrics
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn

from module.dann import DANNModel, DANNLRScheduler


class LitDANN(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()

        self.config = config
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr

        self.model = DANNModel(
            config,
            classes=self.config.num_classes,
        )

        self.init_metrics()

        self.register_buffer(
            "targets_dsc_src", torch.ones(config.batch_size).long())
        self.register_buffer(
            "targets_dsc_tgt", torch.zeros(config.batch_size).long())

    def training_step(self, batch, _):
        inputs_src, targets_src = batch["src"]
        inputs_tgt, _ = batch["tgt"]

        p = self.get_p()
        lambda_p = self.get_lambda_p(p)

        stage = "train"
        outputs_src, outputs_dsc_src, outputs_dsc_tgt = self.model.forward_train(inputs_src, inputs_tgt, lambda_p)
        loss = F.cross_entropy(outputs_src, targets_src)
        acc = self.acc[stage](outputs_src.argmax(dim=-1), targets_src)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

        loss_dsc = F.cross_entropy(outputs_dsc_src, self.targets_dsc_src) + F.cross_entropy(outputs_dsc_tgt, self.targets_dsc_tgt)
        self.log(f"{stage}_loss_dsc", loss_dsc, on_step=False, on_epoch=True, sync_dist=True)
        loss += loss_dsc

        self.log(f"p", p, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"lr", self.optimizers().param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_epoch_end(self, _):
        stage = "train"
        self.log(f"{stage}_acc_epoch", self.acc[stage].compute(),
                 prog_bar=True, logger=True, sync_dist=True)

    def validation_step(self, batch, _):
        inputs_tgt, targets_tgt = batch

        stage = "val"
        outputs_tgt = self.model(inputs_tgt)
        loss = F.cross_entropy(outputs_tgt, targets_tgt)
        acc = self.acc[stage]["tgt"](outputs_tgt.argmax(dim=-1), targets_tgt)

        self.log(f"{stage}_loss_tgt", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc_tgt", acc, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, _):
        stage = "val"
        self.log(f"{stage}_acc_tgt_epoch", self.acc[stage]["tgt"].compute(),
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
        model_params = [
            {
                "params": self.model.feature_extractor.parameters(),
                "lr": self.learning_rate,
                "weight_lr": self.config.w_enc,
            },
            {
                "params": self.model.head.parameters(),
                "lr": self.learning_rate,
                "weight_lr": self.config.w_cls,
            },
            {
                "params": self.model.discriminator.parameters(),
                "lr": self.learning_rate,
                "weight_lr": self.config.w_dsc,
            },
        ]
        if self.config.optimizer == "sgd":
            optimizer = torch.optim.SGD(
            model_params,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-3,
                nesterov=True,
            )
        elif self.config.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model_params,
                lr=self.learning_rate,
            )
        elif self.config.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model_params,
                lr=self.learning_rate,
            )

        if self.config.lr_schedule:
            lr_scheduler = DANNLRScheduler(
                optimizer=optimizer,
                init_lr=self.learning_rate,
                alpha=self.config.alpha,
                beta=self.config.beta,
                total_steps=len(self.train_dataloader()) * self.config.max_epochs
            )
            scheduler = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
            return [optimizer], [scheduler]
        else:
            return optimizer

    def get_p(self):
        current_iterations = self.global_step
        len_dataloader = len(self.train_dataloader())
        p = float(current_iterations /
                (self.config.max_epochs * len_dataloader))

        return p

    def get_lambda_p(self, p):
        lambda_p = 2. / (1. + np.exp(-self.config.gamma * p)) - 1

        return lambda_p

    def init_metrics(self):
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc_src = torchmetrics.Accuracy()
        self.val_acc_tgt = torchmetrics.Accuracy()
        self.test_acc = nn.ModuleList([
            torchmetrics.Accuracy() for _ in range(self.config.num_test_sets)
        ])

        self.acc = {
            "train": self.train_acc,
            "val": {
                "src": self.val_acc_src,
                "tgt": self.val_acc_tgt,
            },
            "test": self.test_acc,
        }