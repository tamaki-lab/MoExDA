import argparse

from tqdm import tqdm

import torch
from torch import nn

from args import ArgParse
from dataset import configure_dataloader, make_class_to_ids
from model import (
    configure_model,
    ModelConfig,
)
from setup import configure_optimizer, configure_scheduler

from logger import configure_logger
from utils import (
    save_to_checkpoint, save_to_comet,
    load_from_checkpoint,
)

from train import train, TrainConfig
from val import validation


class TqdmEpoch(tqdm):
    def __init__(
            self,
            start_epoch: int,
            num_epochs: int,
            *args,
            **kwargs):
        super().__init__(
            range(start_epoch + 1, num_epochs + 1), *args, **kwargs
        )


def cut_strings_in_dict(class_to_ids, substring):
    new_class_to_ids = {}
    for key, value in class_to_ids.items():
        cut_index = key.find(substring)
        if cut_index != -1:
            new_key = key[:cut_index]
        else:
            new_key = key
        new_class_to_ids[new_key] = value
    return new_class_to_ids


def prepare_training(args: argparse.Namespace):
    """prepare training objects from args

    Args:
        args (argparse.Namespace): command line argmentrs

    Returns:
        a set of training objects
    """

    logger = configure_logger(
        logged_params=vars(args),
        model_name=args.model_name,
        disable_logging=args.disable_comet,
    )

    dataloaders = configure_dataloader(
        command_line_args=args,
        dataset_name=args.dataset_name,
    )

    # class_to_ids = dataloaders.train_loader.dataset.find_classes()
    class_to_ids = make_class_to_ids(dataloaders.train_loader, args.cls_id_to_cls_name_file)

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    model = configure_model(ModelConfig(
        model_name=args.model_name,
        use_pretrained=args.use_pretrained,
        torch_home=args.torch_home,
        n_classes=dataloaders.n_classes,
    ), class_to_ids=class_to_ids)
    model = model.to(device)
    if args.use_dp:
        model = nn.DataParallel(model)  # type: ignore[assignment]

    optimizer = configure_optimizer(
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        model_params=model.parameters()
    )
    scheduler = configure_scheduler(
        optimizer=optimizer,
        scheduler=args.scheduler
    )

    train_config = TrainConfig(
        grad_accum_interval=args.grad_accum,
        log_interval_steps=args.log_interval_steps
    )

    if args.checkpoint_to_resume:
        (
            start_epoch,
            current_train_step,
            current_val_step,
            model,
            optimizer,
            scheduler,
        ) = load_from_checkpoint(  # type: ignore[assignment]
            args.checkpoint_to_resume,
            model,
            optimizer,
            scheduler,
            device
        )
    else:
        current_train_step = 1
        current_val_step = 1
        start_epoch = 0

    return (
        logger,
        dataloaders,
        model,
        optimizer,
        scheduler,
        train_config,
        current_train_step,
        current_val_step,
        start_epoch,
    )


class ValidationChecker:
    def __init__(self, val_interval_epochs, num_epochs):
        self.val_interval_epochs = val_interval_epochs
        self.num_epochs = num_epochs

    def should_validate(self, current_epoch):
        return (
            current_epoch % self.val_interval_epochs == 0
            or current_epoch == self.num_epochs
        )


def main():

    args = ArgParse.get()

    (
        logger,
        dataloaders,
        model,
        optimizer,
        scheduler,
        train_config,
        current_train_step,
        current_val_step,
        start_epoch,
    ) = prepare_training(args)

    val_checker = ValidationChecker(args.val_interval_epochs, args.num_epochs)
    max_epoch = 1 if args.val_only else args.num_epochs
    with TqdmEpoch(
        start_epoch, max_epoch, unit='epoch',
    ) as progress_bar_epoch:
        for current_epoch in progress_bar_epoch:
            progress_bar_epoch.set_description(f"[epoch {current_epoch:03d}]")
            if not args.val_only:
                train_output = train(
                    model,
                    optimizer,
                    scheduler,
                    dataloaders.train_loader,
                    current_train_step,
                    current_epoch,
                    logger,
                    train_config
                )
                current_train_step = train_output.train_step

            if val_checker.should_validate(current_epoch):

                val_output = validation(
                    model,
                    dataloaders.val_loader,
                    current_val_step,
                    current_epoch,
                    logger,
                )
                current_val_step = val_output.val_step

                checkpoint_dict, _ = save_to_checkpoint(
                    args.save_checkpoint_dir,
                    current_epoch,
                    current_train_step,
                    current_val_step,
                    val_output.top1,
                    model,
                    optimizer,
                    scheduler,
                    logger
                )
                save_to_comet(
                    checkpoint_dict,
                    args.model_name,
                    logger
                )


if __name__ == "__main__":
    main()
