from typing import Tuple

from comet_ml import Experiment
from lightning.pytorch.loggers import CometLogger
# from args import ArgParse
from .make_expname import make_expname
import argparse


def configure_logger_pl(
        args: argparse.Namespace,
        model_name: str,
        disable_logging: bool,
        save_dir: str,
) -> Tuple[Experiment, str]:
    """comet logger factory

    Args:
        model_name (str): modelname to be added as a tag of comet experiment
        disable_logging (bool): disable comet Experiment object
        save_dir (str): dir to save comet log

    Returns:
        comet_ml.Experiment: logger
        str: experiment name of comet.ml
    """
    # args = ArgParse.get()

    # optimizer_name = args.optimizer_name
    # scheduler = args.scheduler
    # lr = args.lr
    # weight_decay = args.weight_decay
    # dataset_name = args.dataset_name

    # exp_name = (
    #     model_name
    #     + f"_optim_{optimizer_name}_sche_{scheduler}_lr_{lr}_wd_{weight_decay}_dataset_{dataset_name}")
    # exp_name = exp_name.replace(" ", "_")
    exp_name, tags = make_expname(args)

    # Use ./.comet.config and ~/.comet.config
    # to specify API key, workspace and project name.
    # DO NOT put API key in the code!
    comet_logger = CometLogger(
        save_dir=save_dir,
        experiment_name=exp_name,
        parse_args=True,
        disabled=disable_logging,
    )
    comet_logger.experiment.add_tags(tags)

    return comet_logger, exp_name
