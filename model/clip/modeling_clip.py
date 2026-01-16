from model import ModelConfig, ClassificationBaseModel, ModelOutput
import torch
from typing import Optional

from transformers import CLIPTokenizer, CLIPModel


class CLIP(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig, class_to_ids: dict):
        super().__init__(model_config)
        self.prepare_model()
        # コンストラクタから，datasetのラベル名：idの辞書をもらう
        self.class_to_ids = class_to_ids

    def prepare_model(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        device = pixel_values.device

        texts = ['a simple line sketch of a ' + i for i in self.class_to_ids.keys()]

        inputs = self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding=True).to(device)

        outputs = self.model(**inputs, pixel_values=pixel_values)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

        return ModelOutput(
            logits=logits_per_image,
            loss=None,
        )
