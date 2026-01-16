from typing import Literal
from dataclasses import dataclass, field
from typing import Optional


SupportedModels = Literal[
    "resnet18",
    "resnet50",
    "abn_r50",
    "vit_b",
    "x3d",
    "zero_output_dummy",
    "clip",
    "swin_transformer",
    "DeiT",
    "CvT",
    "vivit",
    "video_vit",
    "timeSformer",
    "Moexlayervit"
]


@dataclass
class ModelConfig:
    model_name: SupportedModels = "resnet18"
    use_pretrained: bool = True
    torch_home: str = "./"
    n_classes: int = 10
    checkpoint_path: Optional[str] = None
    moex_layers: Optional[list] = None
    cos_sim: bool = False
    moexda_dict: Optional[dict] = field(default_factory=dict)
