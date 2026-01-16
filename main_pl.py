
import torch
import lightning.pytorch as pl
from lightning.pytorch.plugins import TorchSyncBatchNorm
from pytorch_lightning.strategies import DDPStrategy


from args import ArgParse
from logger import configure_logger_pl
from callback import MyPrintingCallback, MyCosineSimilarityCallback, MytSNECallBack, MyKLDivergenceCallback
from dataset import TrainValDataModule
from model import SimpleLightningModel
from transformers.utils import logging
logging.set_verbosity_error()  # Suppress warnings from transformers library


def main():
    assert torch.cuda.is_available()

    args = ArgParse.get()

    loggers, exp_name = configure_logger_pl(
        args=args,
        model_name=args.model_name,
        disable_logging=args.disable_comet,
        save_dir=args.comet_log_dir,
    )
    data_module = TrainValDataModule(
        command_line_args=args,
        dataset_name=args.dataset_name,
    )
    model_lightning = SimpleLightningModel(
        command_line_args=args,
        n_classes=data_module.n_classes,
        exp_name=exp_name
    )

    # callbacks = [MyPrintingCallback()]
    if args.cos_sim:
        callbacks = [MyCosineSimilarityCallback()]
    # callbacks = [MytSNECallBack()]
    # callbacks = [MyKLDivergenceCallback()]
    else:
        callbacks = []

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    trainer = pl.Trainer(
        # strategy=DDPStrategy(find_unused_parameters=True),
        devices=args.devices,
        accelerator="gpu",
        strategy='ddp_find_unused_parameters_true',
        max_epochs=args.num_epochs,
        logger=loggers,
        log_every_n_steps=args.log_interval_steps,
        accumulate_grad_batches=args.grad_accum,
        num_sanity_val_steps=0,
        precision="bf16-mixed",  # for FP16 training, use with caution for nan/inf
        # precision="bf16-true",
        # fast_dev_run=True, # only for debug
        # fast_dev_run=5,  # only for debug
        # limit_train_batches=15,  # only for debug
        # limit_val_batches=15,  # only for debug
        callbacks=callbacks,
        plugins=[TorchSyncBatchNorm()],
        # profiler="simple",
    )

    if args.val_only:
        trainer.validate(
            model=model_lightning,
            datamodule=data_module,
            ckpt_path=args.checkpoint_to_resume,
        )
    else:
        trainer.fit(
            model=model_lightning,
            datamodule=data_module,
            ckpt_path=args.checkpoint_to_resume,
        )


if __name__ == "__main__":
    main()
