import torch
import torchmetrics
import torchvision
import wandb

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from module.zsgcgan import ZSGcGANModel


class LitZSGCADA(pl.LightningModule):
    def __init__(self, config, learning_rate=None):
        super().__init__()

        self.config =config
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = config.lr

        self.model = ZSGcGANModel(config)

        self.init_metrics()

        self.register_buffer(
            "real_targets", torch.ones(config.batch_size).float())
        self.register_buffer(
            "translated_targets", torch.zeros(config.batch_size).float())
        self.register_buffer(
            "fake_targets", torch.zeros(config.batch_size).float())

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
                       inputs_src_translated_self, outputs_src_toi_self_translated, outputs_tgt_toi_syn, \
                            outputs_dsc_b, outputs_dsc_b_gc, outputs_dsc_c_src, outputs_dsc_c_tgt, \
                                outputs_dsc_b_self_translated_sem_idt = self.model.forward_gan(inputs_src, inputs_tgt)

            loss_gan = self.config.lambda_real * (
                F.mse_loss(outputs_dsc_src_translated, self.real_targets) +
                F.mse_loss(outputs_dsc_src_gc_translated, self.real_targets)
            ) / 2
            self.log(f"{stage}_loss_gan", loss_gan, on_step=False, on_epoch=True, sync_dist=True)
            loss = loss_gan

            loss_gan_enc = self.config.lambda_real_enc * (
                F.mse_loss(outputs_dsc_b, self.fake_targets) +
                F.mse_loss(outputs_dsc_b_gc, self.fake_targets) +
                F.mse_loss(outputs_dsc_c_src, self.fake_targets) +
                F.mse_loss(outputs_dsc_c_tgt, self.fake_targets) +
                F.mse_loss(outputs_dsc_b_self_translated_sem_idt, self.fake_targets)
            ) / 5
            self.log(f"{stage}_loss_gan_enc", loss_gan_enc, on_step=False, on_epoch=True, sync_dist=True)
            loss += loss_gan_enc

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
                    loss_idt_sem = self.config.lambda_sem_idt * (
                        F.cross_entropy(outputs_src_toi_self_translated, targets_src) +
                        F.cross_entropy(outputs_tgt_toi_syn, targets_src)
                    ) / 2
                    loss += loss_idt_sem
                    self.log(f"{stage}_loss_idt_sem", loss_idt_sem, on_step=False, on_epoch=True, sync_dist=True)

            self.log(f"{stage}_loss_g", loss, on_step=False, on_epoch=True, sync_dist=True)
        elif optimizer_idx == 1:
            # discriminator phase
            inputs_src, _ = batch["src"]
            inputs_tgt, _ = batch["tgt"]

            outputs_dsc_src_toi, outputs_dsc_src_irt_translated, \
                outputs_dsc_src_toi_gc, outputs_dsc_src_irt_gc_translated, \
                    outputs_dsc_irt, outputs_dsc_irt_gc, outputs_dsc_toi, \
                        outputs_dsc_tgt, outputs_dsc_tgt_gc, outputs_dsc_src = self.model.forward_dsc(inputs_src, inputs_tgt)

            # GAN loss
            loss_gan = (
                self.config.lambda_real * F.mse_loss(outputs_dsc_src_toi, self.real_targets) +
                self.config.lambda_fake * F.mse_loss(outputs_dsc_src_irt_translated, self.translated_targets)
            ) / 2 + (
                self.config.lambda_real * F.mse_loss(outputs_dsc_src_toi_gc, self.real_targets) +
                self.config.lambda_fake * F.mse_loss(outputs_dsc_src_irt_gc_translated, self.translated_targets)
            ) / 2

            loss = loss_gan

            self.log(f"{stage}_loss_gan_d", loss_gan, on_step=False, on_epoch=True, sync_dist=True)

            loss_gan_enc = self.config.lambda_fake * (
                F.mse_loss(outputs_dsc_irt, self.real_targets) +
                F.mse_loss(outputs_dsc_irt_gc, self.real_targets) +
                F.mse_loss(outputs_dsc_toi, self.real_targets) +
                F.mse_loss(outputs_dsc_tgt, self.real_targets) +
                F.mse_loss(outputs_dsc_tgt_gc, self.real_targets) +
                F.mse_loss(outputs_dsc_src, self.real_targets)
            ) / 6
            loss += loss_gan_enc
            self.log(f"{stage}_loss_gan_enc_d", loss_gan_enc, on_step=False, on_epoch=True, sync_dist=True)

            self.log(f"{stage}_loss_d", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, _):
        inputs_tgt, targets_tgt = batch

        stage = "val"
        translated, _, _ = self.model.generator_ts(inputs_tgt)
        outputs_tgt = self.model(translated)
        loss = F.cross_entropy(outputs_tgt, targets_tgt)
        acc = self.acc[stage]["tgt"](outputs_tgt.argmax(dim=-1), targets_tgt)

        self.log(f"{stage}_loss_tgt", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc_tgt", acc, on_step=False, on_epoch=True, sync_dist=True)

        # generate images
        if self.plt_save_val:
            self.before = inputs_tgt[:10]
            self.after = translated[:10]
            self.plt_save_val = False

        return loss

    def validation_epoch_end(self, _):
        stage = "val"
        self.log(f"{stage}_acc_tgt_epoch", self.acc[stage]["tgt"].compute(),
                 prog_bar=True, logger=True, sync_dist=True)

        self.plt_save_val = True
        before = torchvision.utils.make_grid(self.before)
        self.logger.experiment.log({"before": [wandb.Image(before)]})
        after = torchvision.utils.make_grid(self.after)
        self.logger.experiment.log({"after": [wandb.Image(after)]})


    def test_step(self, batch, _, loader_idx):
        inputs, targets = batch

        stage = "test"
        translated, _, _ = self.model.generator_ts(inputs)
        outputs = self.model(translated)
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
        model_params_g_enc = self.model.get_parameters_generator_enc()
        model_params_d_enc = self.model.get_parameters_discriminator_enc()
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
            optimizer_g_enc = torch.optim.SGD(
                model_params_g_enc,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-3,
                nesterov=True,
            )
            optimizer_d_enc = torch.optim.SGD(
                model_params_d_enc,
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-3,
                nesterov=True,
            )
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
            optimizer_g_enc = torch.optim.Adam(
                model_params_g_enc,
                lr=self.learning_rate,
                betas=(self.config.beta1, 0.999),
            )
            optimizer_d_enc = torch.optim.Adam(
                model_params_d_enc,
                lr=self.learning_rate,
                betas=(self.config.beta1, 0.999),
            )
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
            optimizer_g_enc = torch.optim.AdamW(
                model_params_g_enc,
                lr=self.learning_rate,
                betas=(self.config.beta1, 0.999),
            )
            optimizer_d_enc = torch.optim.AdamW(
                model_params_d_enc,
                lr=self.learning_rate,
                betas=(self.config.beta1, 0.999),
            )

        return [optimizer_g, optimizer_d, optimizer_g_enc, optimizer_d_enc]

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
