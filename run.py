from omegaconf import open_dict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import hydra
import pytorch_lightning as pl

from dataset.nexmon import NexmonDataModule
from dataset.digit import DigitDataModule
from pl_module.timm import LitTimm
from pl_module.dann import LitDANN
from pl_module.adda import LitADDA
from pl_module.cycada import LitCyCADA
from pl_module.gcada import LitGCADA
from pl_module.zsgcada import LitZSGCADA


@hydra.main(config_path=".", config_name="common")
def run(config):
    pl.seed_everything(config.seed)

    logger = None
    if config.logger:
        task = config.root.split("/")[-1].split("_")[0]
        from pytorch_lightning.loggers import WandbLogger
        logger = WandbLogger(
            project=config.project,
            name=f"{config.model_name}-{config.backbone_name}-{task}",
            config=config,
        )

    callbacks = []
    if config.es_patience is not None:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=config.es_patience,
            verbose=False,
            mode='min'
        ))

    if config.task == "nexmon":
        datamodule = NexmonDataModule(config)
        with open_dict(config):
            config.num_classes = datamodule.num_classes
            config.num_test_sets = datamodule.num_test_sets
    elif config.task == "digit":
        datamodule = DigitDataModule(config)
        with open_dict(config):
            config.num_classes = datamodule.num_classes
            config.num_test_sets = 4

    if config.model_name == "timm":
        model = LitTimm(config)
        callbacks.append(ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename=f"{config.model_name}-{config.backbone_name}-"+"{epoch:02d}-".replace("=","") +
                f"task={config.task}-"+"{val_loss:.4f}".replace("=",""),
            monitor="val_loss",
            mode="min"
        ))
    elif config.model_name == "dann":
        model = LitDANN(config)
        callbacks.append(ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename=f"{config.model_name}-{config.backbone_name}-"+"{epoch:02d}-".replace("=","") +
                f"task={config.task}-"+"{val_loss_tgt:.4f}".replace("=",""),
            save_top_k=-1,
        ))
    elif config.model_name == "adda":
        model = LitADDA(config)
    elif config.model_name == "cycada":
        model = LitCyCADA(config)
        # callbacks.append(ModelCheckpoint(
        #     dirpath=config.checkpoint_dir,
        #     filename=f"{config.model_name}-{config.backbone_name}-"+"{epoch:02d}-".replace("=","") +
        #         f"task={config.task}-"+"{val_loss:.4f}".replace("=",""),
        #     save_top_k=-1,
        # ))
    elif config.model_name == "gcada":
        model = LitGCADA(config)
        # callbacks.append(ModelCheckpoint(
        #     dirpath=config.checkpoint_dir,
        #     filename=f"{config.model_name}-{config.backbone_name}-"+"{epoch:02d}-".replace("=","") +
        #         f"task={config.task}-"+"{val_loss_tgt:.4f}".replace("=",""),
        #     save_top_k=-1,
        # ))
    elif config.model_name == "zsgcada":
        model = LitZSGCADA(config)
        # callbacks.append(ModelCheckpoint(
        #     dirpath=config.checkpoint_dir,
        #     filename=f"{config.model_name}-{config.backbone_name}-"+"{epoch:02d}-".replace("=","") +
        #         f"task={config.task}-"+"{val_loss_tgt:.4f}".replace("=",""),
        #     save_top_k=-1,
        # ))

    trainer = pl.Trainer(
        precision=16,
        callbacks=callbacks,
        accumulate_grad_batches=config.grad_accum,
        deterministic=True,
        check_val_every_n_epoch=1,
        gpus=config.gpus,
        logger=logger,
        max_epochs=config.max_epochs,
        weights_summary="top",
        accelerator='ddp',
    )
    trainer.fit(model, datamodule=datamodule)
    if config.task != "nexmon":
        trainer.test()


if __name__ == '__main__':
    run()