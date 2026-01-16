from typing import Optional

import torch

from transformers import ViTForImageClassification, ViTConfig

from model import ModelConfig, ClassificationBaseModel, ModelOutput

from torch.nn import CrossEntropyLoss


class ViTbv(ClassificationBaseModel):

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.model_config = model_config
        self.prepare_model()

    def prepare_model(self):
        if self.model_config.use_pretrained:
            self.model = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=self.model_config.n_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            vit_config = ViTConfig(
                num_labels=self.model_config.n_classes,
            )
            self.model = ViTForImageClassification(
                vit_config,
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:

        B, T, C, H, W = pixel_values.shape  # pixel_values : (B,T,C,H,W)
        pixel_values = pixel_values.reshape(-1, C, H, W)  # (B,T,C,H,W) -> (B*T,C,H,W)

        logits = 0
        num_labels = self.model_config.n_classes

        output = self.model(pixel_values)
        logits += output.logits

        logits = logits.view(B, T, -1)
        logits = logits.mean(dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss = None

        return ModelOutput(
            logits=logits,
            loss=loss
        )
