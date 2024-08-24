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
                       inputs_src_translated_self, outputs_src_self_translated, outputs_src, \
                        outputs_rec_src, outputs_rec_tgt, \
                            outputs_syn_src, outputs_syn_tgt = self.model.forward_gan(inputs_src, inputs_tgt)

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


            loss_cross = self.config.lambda_cross * (
                F.mse_loss(outputs_rec_src, self.fake_targets) +
                F.mse_loss(outputs_rec_tgt, self.fake_targets) +
                F.mse_loss(outputs_syn_src, self.real_targets) +
                F.mse_loss(outputs_syn_tgt, self.real_targets)
            ) / 2

            self.log(f"{stage}_loss_g_up", loss, on_step=False, on_epoch=True, sync_dist=True)
            loss += loss_cross

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

            outputs_dsc_src_toi, outputs_dsc_src_irt_translated, \
                outputs_dsc_src_toi_gc, outputs_dsc_src_irt_gc_translated, \
                outputs_rec_src, outputs_rec_tgt, \
                    outputs_syn_src, outputs_syn_tgt = self.model.forward_dsc(inputs_src, inputs_tgt)

            # GAN loss
            loss_gan = (
                self.config.lambda_real * F.mse_loss(outputs_dsc_src_toi, self.real_targets) +
                self.config.lambda_fake * F.mse_loss(outputs_dsc_src_irt_translated, self.translated_targets)
            ) / 2 + (
                self.config.lambda_real * F.mse_loss(outputs_dsc_src_toi_gc, self.real_targets) +
                self.config.lambda_fake * F.mse_loss(outputs_dsc_src_irt_gc_translated, self.translated_targets)
            ) / 2

            loss = loss_gan
            self.log(f"{stage}_loss_d", loss, on_step=False, on_epoch=True, sync_dist=True)

            loss_cross = self.config.lambda_cross * (
                F.mse_loss(outputs_rec_src, self.real_targets) +
                F.mse_loss(outputs_rec_tgt, self.real_targets) +
                F.mse_loss(outputs_syn_src, self.fake_targets) +
                F.mse_loss(outputs_syn_tgt, self.fake_targets)
            ) / 2

            self.log(f"{stage}_loss_d_up", loss, on_step=False, on_epoch=True, sync_dist=True)
            loss += loss_cross

        elif optimizer_idx == 2:
            # content generator
            inputs_src, targets_src = batch["src"]
            inputs_tgt, _ = batch["tgt"]

            outputs_dsc_d_src, outputs_dsc_d_tgt = self.model.forward_enc_c(inputs_src, inputs_tgt)

            loss = self.config.lambda_enc * (
                F.mse_loss(outputs_dsc_d_src, self.real_targets) +
                F.mse_loss(outputs_dsc_d_tgt, self.fake_targets)
            ) / 2
            self.log(f"{stage}_loss_g_enc_c", loss, on_step=False, on_epoch=True, sync_dist=True)

        elif optimizer_idx == 3:
        # elif optimizer_idx == 2:
            # content discriminator
            inputs_src, targets_src = batch["src"]
            inputs_tgt, _ = batch["tgt"]

            outputs_dsc_d_src, outputs_dsc_d_tgt = self.model.forward_enc_c(inputs_src, inputs_tgt)

            loss = self.config.lambda_enc * (
                F.mse_loss(outputs_dsc_d_src, self.fake_targets) +
                F.mse_loss(outputs_dsc_d_tgt, self.real_targets)
            ) / 2
            self.log(f"{stage}_loss_d_enc_c", loss, on_step=False, on_epoch=True, sync_dist=True)

        elif optimizer_idx == 4:
            # background generator
            inputs_src, targets_src = batch["src"]
            inputs_tgt, _ = batch["tgt"]

            outputs_dsc_c_toi, outputs_dsc_c_irt = self.model.forward_enc_b(inputs_src, inputs_tgt)

            loss = self.config.lambda_enc * (
                F.mse_loss(outputs_dsc_c_toi, self.real_targets) +
                F.mse_loss(outputs_dsc_c_irt, self.fake_targets)
            ) / 2
            self.log(f"{stage}_loss_g_enc_b", loss, on_step=False, on_epoch=True, sync_dist=True)

        elif optimizer_idx == 5:
        # elif optimizer_idx == 3:
            # background discriminator
            inputs_src, targets_src = batch["src"]
            inputs_tgt, _ = batch["tgt"]

            outputs_dsc_c_toi, outputs_dsc_c_irt = self.model.forward_enc_b(inputs_src, inputs_tgt)

            loss = self.config.lambda_enc * (
                F.mse_loss(outputs_dsc_c_toi, self.fake_targets) +
                F.mse_loss(outputs_dsc_c_irt, self.real_targets)
            ) / 2
            self.log(f"{stage}_loss_d_enc_b", loss, on_step=False, on_epoch=True, sync_dist=True)
        # elif optimizer_idx == 6:
        #     inputs_src, targets_src = batch["src"]
        #     inputs_tgt, _ = batch["tgt"]

        #     outputs_rec_src, outputs_rec_tgt, \
        #         outputs_cross_src, outputs_cross_tgt = self.model.forward_cross(inputs_src, inputs_tgt)

        #     loss = self.config.lambda_cross * (
        #         F.mse_loss(outputs_rec_src, self.fake_targets) +
        #         F.mse_loss(outputs_rec_tgt, self.fake_targets) +
        #         F.mse_loss(outputs_cross_src, self.real_targets) +
        #         F.mse_loss(outputs_cross_tgt, self.real_targets)
        #     ) / 2

        #     self.log(f"{stage}_loss_g_up", loss, on_step=False, on_epoch=True, sync_dist=True)
        # elif optimizer_idx == 7:
        #     inputs_src, targets_src = batch["src"]
        #     inputs_tgt, _ = batch["tgt"]

        #     outputs_rec_src, outputs_rec_tgt, \
        #         outputs_cross_src, outputs_cross_tgt = self.model.forward_cross(inputs_src, inputs_tgt)

        #     loss = self.config.lambda_cross * (
        #         F.mse_loss(outputs_rec_src, self.real_targets) +
        #         F.mse_loss(outputs_rec_tgt, self.real_targets) +
        #         F.mse_loss(outputs_cross_src, self.fake_targets) +
        #         F.mse_loss(outputs_cross_tgt, self.fake_targets)
        #     ) / 2

        #     self.log(f"{stage}_loss_d_up", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, _):
        inputs_tgt, targets_tgt = batch

        stage = "val"
        translated, _, _ = self.model.generator_ts(inputs_tgt)
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
        model_params_enc_c_generator = self.model.get_parameters_generator_enc(enc_type="c")
        model_params_enc_c_discriminator = self.model.get_parameters_discriminator_enc(enc_type="c")
        model_params_enc_b_generator = self.model.get_parameters_generator_enc(enc_type="b")
        model_params_enc_b_discriminator = self.model.get_parameters_discriminator_enc(enc_type="b")
        # model_params_up_generator = self.model.get_parameters_generator_up()
        # model_params_up_discriminator = self.model.get_parameters_discriminator_up()

        optimizer_g = torch.optim.Adam(
            model_params_g,
            lr=self.learning_rate,
            betas=(self.config.beta1, 0.999),
        )
        optimizer_d = torch.optim.Adam(
            model_params_d,
            lr=self.learning_rate * self.config.lr_dsc_multi,
            betas=(self.config.beta1, 0.999),
        )
        optimizer_enc_c_g = torch.optim.Adam(
            model_params_enc_c_generator,
            lr=self.learning_rate * self.config.lr_enc_multi,
            betas=(self.config.beta1, 0.999),
        )
        optimizer_enc_c_d = torch.optim.Adam(
            model_params_enc_c_discriminator,
            lr=self.learning_rate * self.config.lr_enc_multi,
            betas=(self.config.beta1, 0.999),
        )
        optimizer_enc_b_g = torch.optim.Adam(
            model_params_enc_b_generator,
            lr=self.learning_rate * self.config.lr_enc_multi,
            betas=(self.config.beta1, 0.999),
        )
        optimizer_enc_b_d = torch.optim.Adam(
            model_params_enc_b_discriminator,
            lr=self.learning_rate * self.config.lr_enc_multi,
            betas=(self.config.beta1, 0.999),
        )
        # optimizer_up_g = torch.optim.Adam(
        #     model_params_up_generator,
        #     lr=self.learning_rate * self.config.lr_enc_multi,
        #     betas=(self.config.beta1, 0.999),
        # )
        # optimizer_up_d = torch.optim.Adam(
        #     model_params_up_discriminator,
        #     lr=self.learning_rate * self.config.lr_enc_multi,
        #     betas=(self.config.beta1, 0.999),
        # )

        return [
            optimizer_g,
            optimizer_d,
            optimizer_enc_c_g,
            optimizer_enc_c_d,
            optimizer_enc_b_g,
            optimizer_enc_b_d,
            # optimizer_up_g,
            # optimizer_up_d,
        ]


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
