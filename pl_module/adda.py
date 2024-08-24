import torch
import torchmetrics
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

from module.adda import ADDAModel


class LitADDA(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()

        self.config = config
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr

        self.model = ADDAModel(
            config
        )

        self.init_metrics()

        self.register_buffer(
            "targets_dsc_real", torch.ones(config.batch_size).long())
        self.register_buffer(
            "targets_dsc_fake", torch.zeros(config.batch_size).long())

    def training_step(self, batch, _, optimizer_idx):
        stage = "train"
        if optimizer_idx == 0:
            # generator (target feature extractor) phase
            inputs_tgt, _ = batch["tgt"]

            outputs_dsc_tgt = self.model.forward_gan(inputs_tgt)

            loss = F.cross_entropy(outputs_dsc_tgt, self.targets_dsc_real)
            self.log(f"{stage}_loss_g", loss, on_step=False, on_epoch=True, sync_dist=True)
        else:
            # discriminator phase
            inputs_src, _ = batch["src"]
            inputs_tgt, _ = batch["tgt"]

            outputs_dsc_src, outputs_dsc_tgt = self.model.forward_dsc(inputs_src, inputs_tgt)

            loss = F.cross_entropy(outputs_dsc_src, self.targets_dsc_real) + F.cross_entropy(outputs_dsc_tgt, self.targets_dsc_fake)
            self.log(f"{stage}_loss_d", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

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
        model_params_g = self.model.feature_extractor_tgt.parameters()
        model_params_d = self.model.discriminator.parameters()
        if self.config.optimizer == "sgd":
            optimizer_g = torch.optim.SGD(
                model_params_g,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-3,
                nesterov=True,
            )
            optimizer_d = torch.optim.SGD(
                model_params_d,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-3,
                nesterov=True,
            )
        elif self.config.optimizer == "adam":
            optimizer_g = torch.optim.Adam(
                model_params_g,
                lr=self.learning_rate,
            )
            optimizer_d = torch.optim.Adam(
                model_params_d,
                lr=self.learning_rate,
            )
        elif self.config.optimizer == "adamw":
            optimizer_g = torch.optim.AdamW(
                model_params_g,
                lr=self.learning_rate,
            )
            optimizer_d = torch.optim.AdamW(
                model_params_d,
                lr=self.learning_rate,
            )

        return [optimizer_g, optimizer_d]

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
