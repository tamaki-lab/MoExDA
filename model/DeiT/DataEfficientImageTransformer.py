from typing import Optional

import torch

from transformers import DeiTForImageClassification, DeiTConfig

from model import ModelConfig, ClassificationBaseModel, ModelOutput


class DeiT(ClassificationBaseModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.model_config = model_config
        self.prepare_model()

    def prepare_model(self):
        if self.model_config.use_pretrained:
            self.model = DeiTForImageClassification.from_pretrained(
                "facebook/deit-base-distilled-patch16-224",
                num_labels=self.model_config.n_classes,
                ignore_mismatched_sizes=True
            )
        else:
            deit_config = DeiTConfig(
                num_labels=self.model_config.n_classes,
            )
            self.model = DeiTForImageClassification(
                deit_config,
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:

        output = self.model(
            pixel_values=pixel_values,
            labels=labels,
        )

        return ModelOutput(
            logits=output.logits,
            loss=output.loss
        )
