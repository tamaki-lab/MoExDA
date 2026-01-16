import argparse


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter
):
    """show default values of argparse.
    see
    https://stackoverflow.com/questions/18462610/argumentparser-epilog-and-description-formatting-in-conjunction-with-argumentdef
    for details.
    """


class ArgParse:

    @staticmethod
    def get() -> argparse.Namespace:
        """generate argparse object

        Returns:
            args (argparse.Namespace): object of command line arguments
        """
        parser = argparse.ArgumentParser(
            description="simple image/video classification",
            formatter_class=CustomFormatter
        )

        # dataset
        parser.add_argument(
            "-r",
            "--root",
            type=str,
            default="./downloaded_data",
            help="root of dataset.",
        )

        parser.add_argument(
            "-sp",
            "--shards_path",
            type=str,
            default=None,
            help="shards_path of dataset.",
        )

        parser.add_argument(
            "-cp",
            "--checkpoint_path",
            type=str,
            default=None,
            help="path of checkpoint.",
        )

        parser.add_argument(
            "-mp",
            "--mimetics_path",
            type=str,
            default=None,
            help="if use_mimetics is True, mimetics_path of dataset.",
        )

        parser.add_argument(
            "-d",
            "--dataset_name",
            type=str,
            default="CIFAR10",
            choices=[
                "CIFAR10",
                "CIFAR100",
                "ImageFolder",
                "VideoFolder",
                "ZeroImages",
                "Mimetics_wds",
                "UCF101_wds",
                "HMDB51_wds",
                "Kinetics400_wds",
                "Kinetics400_hat",
                "UCF_hat",
                "mim_hat",],
            help="name of dataset.",
        )
        parser.add_argument(
            "-td",
            "--train_dir",
            type=str,
            default="train",
            help="subdir name from root for training set.",
        )
        parser.add_argument(
            "-vd",
            "--val_dir",
            type=str,
            default="val",
            help="subdir name from root for validation set.",
        )

        parser.add_argument(
            "--use_edge",
            dest="use_edge",
            action="store_true",
            help="apply an edge filter to the dataset",
        )

        parser.add_argument(
            "-shuffle",
            "--shuffle_buffer_size",
            type=int,
            default=100,
            help="shuffle buffer size.",
        )

        # model
        parser.add_argument(
            "--torch_home",
            type=str,
            default="./pretrained_models",
            help="TORCH_HOME environment variable "
            "where pre-trained model weights are stored.",
        )
        parser.add_argument(
            "-m",
            "--model_name",
            type=str,
            default="resnet18",
            choices=[
                "resnet18",
                "resnet50",
                "x3d",
                "abn_r50",
                "vit_b",
                "zero_output_dummy",
                "clip",
                "swin_transformer",
                "DeiT",
                "CvT",
                "vivit",
                "video_vit",
                "timeSformer",
                "Moexlayervit",],
            help="name of the model",
        )

        parser.add_argument(
            "--use_pretrained",
            dest="use_pretrained",
            action="store_true",
            help="use pretrained model weights (default)",
        )
        parser.add_argument(
            "--scratch",
            dest="use_pretrained",
            action="store_false",
            help="do not use pretrained model weights, "
            "instead train from scratch (not default)",
        )
        parser.set_defaults(use_pretrained=True)

        # video
        parser.add_argument(
            "--frames_per_clip",
            type=int,
            default=16,
            help="frames per clip."
        )
        parser.add_argument(
            "--clip_duration",
            type=float,
            default=80 / 30,
            help="duration of a clip (in second).",
        )
        parser.add_argument(
            "--clips_per_video",
            type=int,
            default=1,
            help="sampling clips per video for validation",
        )

        # training
        parser.add_argument(
            "-b",
            "--batch_size",
            type=int,
            default=8,
            help="batch size."
        )
        parser.add_argument(
            "-w",
            "--num_workers",
            type=int,
            default=2,
            help="number of workers."
        )
        parser.add_argument(
            "-e",
            "--num_epochs",
            type=int,
            default=25,
            help="number of epochs."
        )
        parser.add_argument(
            "-vi",
            "--val_interval_epochs",
            type=int,
            default=1,
            help="validation interval in epochs.",
        )
        parser.add_argument(
            "-li",
            "--log_interval_steps",
            type=int,
            default=1,
            help="logging interval in steps.",
        )

        # optimizer
        parser.add_argument(
            "--optimizer_name",
            type=str,
            default="SGD",
            choices=["SGD", "Adam", "AdamW"],
            help="optimizer name.",
        )
        parser.add_argument(
            "--grad_accum",
            type=int,
            default=1,
            help="steps to accumlate gradients.",
        )
        parser.add_argument(
            "-lr",
            type=float,
            default=1e-4,
            help="learning rate."
        )
        parser.add_argument(
            "--momentum",
            type=float,
            default=0.9,
            help="momentum of SGD."
        )
        parser.add_argument(
            "--weight_decay",
            type=float,
            default=5e-4,
            help="weight decay."
        )
        # parser.add_argument(
        #     "--use_scheduler",
        #     dest="use_scheduler",
        #     action="store_true",
        #     help="use scheduler (not default)",
        # )
        parser.add_argument(
            "--scheduler",
            type=str,
            default=None,
            choices=['None', 'CosineAnnealingLR', 'StepLR', 'CosineAnnealingWarmRestarts'],
            help="Learning rate scheduler. Choices are 'None', 'CosineAnnealingLR', and 'StepLR'."
        )
        parser.add_argument(
            "--no_scheduler",
            dest="use_scheduler",
            action="store_false",
            help="do not use scheduler (default)",
        )
        parser.set_defaults(use_scheduler=False)

        parser.add_argument(
            "--use_moex",
            dest="use_moex",
            action="store_true",
            help="use moment exchange (MoEx)",
        )

        parser.add_argument(
            "--use_hat",
            dest="use_hat",
            action="store_true",
            help="use moment exchange (MoEx)",
        )

        parser.add_argument(
            "-ml",
            "--moex_layers",
            type=int,
            nargs="+",
            default=None,
            help="choose layers for MoEx.",
        )

        parser.add_argument(
            "-norm",
            "--norm_type",
            type=str,
            default="pono",
            choices=["in", "pono"],
            help="choose layers for MoEx.",
        )

        parser.add_argument(
            "-pos_moex",
            "--position_moex",
            type=str,
            default="AfterMHA",
            choices=["BeforeMHA", "AfterMHA", "BeforeMLP", "AfterMLP", "AfterResidual"],
            help="choose layers for MoEx.",
        )
        parser.add_argument(
            "-ex_direction",
            "--exchange_direction",
            type=str,
            default="edge_to_rgb",
            choices=["edge_to_rgb", "rgb_to_edge", "bidirectional"],
            help="choose direction for MoEx.",
        )

        parser.add_argument(
            "-stop_grad",
            "--stop_gradient",
            type=bool,
            default=False,
            choices=[True, False],
            help="stop gradient for MoEx.",
        )

        # multi-GPU strategy
        parser.add_argument(
            "--use_dp",
            dest="use_dp",
            action="store_true",
            help="GPUs with data parallel (dp); not for lightning",
        )
        parser.set_defaults(use_dp=False)

        parser.add_argument(
            "--devices",
            "--gpu_ids",
            type=str,
            default="-1",
            help="GPU ID used for ddp strategy (only for lightning)."
            "\"--devices=0,1\" for 0 and 1, \"--devices=-1\" for all gpus (default).",
            # https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html#choosing-gpu-devices
        )

        # log dirs
        parser.add_argument(
            "--comet_log_dir",
            type=str,
            default="./comet_logs/",
            help="dir to comet log files.",
        )
        parser.add_argument(
            "--tf_log_dir",
            type=str,
            default="./tf_logs/",
            help="dir to TensorBoard log files.",
        )

        # checkpoint files
        parser.add_argument(
            "--save_checkpoint_dir",
            type=str,
            default="./log",
            help="dir to save checkpoint files.",
        )
        parser.add_argument(
            "--checkpoint_to_resume",
            type=str,
            default=None,
            help="path to the checkpoint file to resume from.",
        )

        # disabling comet for debugging
        parser.add_argument(
            "--disable_comet",
            "--no_comet",
            dest="disable_comet",
            action="store_true",
            help="do not use comet.ml (default: use comet)",
        )

        parser.add_argument(
            "--val_only",
            action="store_true",
            help="only validate (not default)",
        )

        parser.add_argument(
            "--cos_sim",
            action="store_true",
            help="use cosine similarity callback (not default)",
        )

        parser.set_defaults(disable_comet=False)

        args = parser.parse_args()

        return args
