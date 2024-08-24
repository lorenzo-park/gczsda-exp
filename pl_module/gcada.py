import torch
import torchmetrics
import torchvision
import wandb

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from module.gcgan import GcGANModel


class LitGCADA(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()

        self.config =config
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr

        self.model = GcGANModel(config)

        self.init_metrics()

        self.register_buffer(
            "real_targets", torch.ones(config.batch_size).float())
        self.register_buffer(
            "translated_targets", torch.zeros(config.batch_size).float())

        # Semantic loss threshold
        self.loss_cls = np.inf

        # For visualizations
        self.plt_save_val = True

        self.std_loss = STDLoss()

    def training_step(self, batch, _, optimizer_idx):
        stage = "train"
        if optimizer_idx == 0:
            # generator phase
            inputs_src, targets_src = batch["src"]
            inputs_tgt, _ = batch["tgt"]

            inputs_src_translated, inputs_src_gc_translated_rot, \
                inputs_src_gc_translated, inputs_src_translated_rot, \
                   outputs_dsc_src_translated, outputs_dsc_src_gc_translated, \
                       inputs_src_translated_self, outputs_src_self_translated, outputs_src = self.model.forward_gan(inputs_src, inputs_tgt)

            loss_gan = self.config.lambda_real * (
                F.mse_loss(outputs_dsc_src_translated, self.real_targets) +
                F.mse_loss(outputs_dsc_src_gc_translated, self.real_targets)
            ) / 2
            self.log(f"{stage}_loss_gan", loss_gan, on_step=False, on_epoch=True, sync_dist=True)
            loss = loss_gan

            loss_gc = self.config.lambda_gc * (
                F.l1_loss(inputs_src_translated, inputs_src_gc_translated_rot) +
                F.l1_loss(inputs_src_gc_translated, inputs_src_translated_rot)
            ) / 2
            self.log(f"{stage}_loss_gc", loss_gc, on_step=False, on_epoch=True, sync_dist=True)

            loss += loss_gc

            if self.config.lambda_idt > 0:
                loss_idt = self.config.lambda_idt * F.l1_loss(inputs_src, inputs_src_translated_self)
                loss += loss_idt
                self.log(f"{stage}_loss_idt", loss_idt, on_step=False, on_epoch=True, sync_dist=True)

            if self.config.lambda_sem_idt > 0:
                if self.current_epoch % self.config.sem_idt_per_epoch == 0:
                    loss_idt_sem = self.config.lambda_sem_idt * F.l1_loss(outputs_src_self_translated, outputs_src)
                    loss += loss_idt_sem
                    self.log(f"{stage}_loss_idt_sem", loss_idt_sem, on_step=False, on_epoch=True, sync_dist=True)

            self.log(f"{stage}_loss_g", loss, on_step=False, on_epoch=True, sync_dist=True)
        elif optimizer_idx == 1:
            # discriminator phase
            inputs_src, _ = batch["src"]
            inputs_tgt, _ = batch["tgt"]

            outputs_dsc_src, outputs_dsc_src_translated, \
                outputs_dsc_src_gc, outputs_dsc_src_gc_translated = self.model.forward_dsc(inputs_src, inputs_tgt)

            # GAN loss
            loss_gan = (
                self.config.lambda_real * F.mse_loss(outputs_dsc_src, self.real_targets) +
                self.config.lambda_fake * F.mse_loss(outputs_dsc_src_translated, self.translated_targets)
            ) / 2 + (
                self.config.lambda_real * F.mse_loss(outputs_dsc_src_gc, self.real_targets) +
                self.config.lambda_fake * F.mse_loss(outputs_dsc_src_gc_translated, self.translated_targets)
            ) / 2

            loss = loss_gan

            self.log(f"{stage}_loss_d", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, _):
        inputs_tgt, targets_tgt = batch

        stage = "val"
        translated = self.model.generator_ts(inputs_tgt)
        outputs_tgt_so = self.model(inputs_tgt)
        outputs_tgt = self.model(translated)
        loss = F.cross_entropy(outputs_tgt, targets_tgt)

        self.log(f"{stage}_loss_tgt", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc_so", self.acc[stage]["tgt_so"](
            outputs_tgt_so.argmax(dim=-1), targets_tgt
        ), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc_translated", self.acc[stage]["tgt_translated"](
            outputs_tgt.argmax(dim=-1), targets_tgt
        ), on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc_combined", self.acc[stage]["tgt_combined"](
            (torch.sigmoid(outputs_tgt_so)+torch.sigmoid(outputs_tgt)).argmax(dim=-1), targets_tgt
        ), on_step=False, on_epoch=True, sync_dist=True)

        # generate images
        if self.plt_save_val:
            self.before = inputs_tgt[:10]
            self.after = translated[:10]
            self.plt_save_val = False

        return loss

    def validation_epoch_end(self, _):
        stage = "val"
        self.log(f"{stage}_acc_tgt_so_epoch", self.acc[stage]["tgt_so"].compute(),
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_acc_tgt_translated_epoch", self.acc[stage]["tgt_translated"].compute(),
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_acc_tgt_combined_epoch", self.acc[stage]["tgt_combined"].compute(),
                 prog_bar=True, logger=True, sync_dist=True)

        self.plt_save_val = True
        before = torchvision.utils.make_grid(self.before, nrow=4)
        self.logger.experiment.log({"before": [wandb.Image(before)]})
        after = torchvision.utils.make_grid(self.after, nrow=4)
        self.logger.experiment.log({"after": [wandb.Image(after)]})


    def test_step(self, batch, _, loader_idx):
        inputs, targets = batch

        stage = "test"
        outputs = self.model(self.model.generator_ts(inputs))
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
        model_params_g = self.model.get_parameters_generator()
        model_params_d = self.model.get_parameters_discriminator()
        model_params_cls = self.model.get_parameters_pretrained()
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
            # optimizer_cls = torch.optim.SGD(
            #     model_params_cls,
            #     lr=self.learning_rate,
            #     momentum=0.9,
            #     weight_decay=1e-3,
            #     nesterov=True,
            # )
        elif self.config.optimizer == "adam":
            optimizer_g = torch.optim.Adam(
                model_params_g,
                lr=self.learning_rate,
                betas=(self.config.beta1, 0.999),
            )
            optimizer_d = torch.optim.Adam(
                model_params_d,
                lr=self.learning_rate,
                betas=(self.config.beta1, 0.999),
            )
            # optimizer_cls = torch.optim.Adam(
            #     model_params_cls,
            #     lr=self.learning_rate,
            # )
        elif self.config.optimizer == "adamw":
            optimizer_g = torch.optim.AdamW(
                model_params_g,
                lr=self.learning_rate,
                betas=(self.config.beta1, 0.999),
            )
            optimizer_d = torch.optim.AdamW(
                model_params_d,
                lr=self.learning_rate,
                betas=(self.config.beta1, 0.999),
            )
            # optimizer_cls = torch.optim.AdamW(
            #     model_params_cls,
            #     lr=self.learning_rate,
            # )

        # return [optimizer_g, optimizer_d, optimizer_cls]
        return [optimizer_g, optimizer_d]

    # def optimizer_step(
    #     self,
    #     _,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure,
    #     on_tpu=False,
    #     using_native_amp=False,
    #     using_lbfgs=False,
    # ):
    #     # update generator every step
    #     if optimizer_idx == 0:
    #         optimizer.step(closure=optimizer_closure)

    #     # update discriminator every 2 steps
    #     if optimizer_idx == 1:
    #         if (batch_idx + 1) % 2 == 0:
    #             # the closure (which includes the `training_step`) will be executed by `optimizer.step`
    #             optimizer.step(closure=optimizer_closure)
    #         else:
    #             # call the closure by itself to run `training_step` + `backward` without an optimizer step
    #             optimizer_closure()

    def init_metrics(self):
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc_tgt = torchmetrics.Accuracy()
        self.val_acc_tgt_translated = torchmetrics.Accuracy()
        self.val_acc_tgt_combined = torchmetrics.Accuracy()
        self.test_acc = nn.ModuleList([
            torchmetrics.Accuracy() for _ in range(self.config.num_test_sets)
        ])

        self.acc = {
            "train": self.train_acc,
            "val": {
                "tgt_so": self.val_acc_tgt,
                "tgt_translated": self.val_acc_tgt_translated,
                "tgt_combined": self.val_acc_tgt_combined,
            },
            "test": self.test_acc,
        }

    def get_p(self):
        current_iterations = self.global_step
        len_dataloader = len(self.train_dataloader())
        total_iterations = self.config.max_epochs * len_dataloader
        p = float(current_iterations / total_iterations)

        return p


class STDLoss(nn.Module):
    def __init__(self):
        super(STDLoss, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=-1).std(dim=-1).mean()
