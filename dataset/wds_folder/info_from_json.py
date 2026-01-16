from pathlib import Path
import json
import re


def info_from_json(shard_path):
    json_file = Path(shard_path).glob("*.json")
    json_file = str(next(json_file))  # get the first json file
    with open(json_file, "r") as f:
        info_dic = json.load(f)
    dataset_size = info_dic["dataset size"]
    n_classes = info_dic["n_classes"]
    class_to_index = info_dic["class_to_index"]
    return dataset_size, n_classes, class_to_index


def all_category_texts(shard_path):
    _, _, class_to_index = info_from_json(shard_path)
    all_text = [s for s in class_to_index.keys()]
    return all_text


def category_indices(shard_path):
    _, _, class_to_index = info_from_json(shard_path)
    return class_to_index.values()


def pascal_case_to_natural_sentence(s: str) -> str:
    """Convert string pascal case to natural sentence.(HelloWorldPython -> Hello world python)"""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", s).replace("[", "").replace("]", "") if s not in ["YoYo"] else s
