import argparse
from typing import Tuple


def make_expname(args: argparse.Namespace) -> Tuple[str, list]:
    exp_name = f"{args.moex_layers}_{args.dataset_name.rstrip("_wds")}"

    exp_name = exp_name.replace(" ", "_")
    tags = []
    tags.append(args.model_name)
    tags.append(args.dataset_name.rstrip("_wds"))
    if args.use_edge:
        tags.append("edge")

    return exp_name, tags
