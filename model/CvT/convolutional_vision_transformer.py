from typing import Optional

from transformers import CvtConfig, CvtForImageClassification
import torch
from model import ModelConfig, ClassificationBaseModel, ModelOutput


class Cvt(ClassificationBaseModel):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.model_config = model_config
        self.prepare_model()

    def prepare_model(self):
        if self.model_config.use_pretrained:
            self.model = CvtForImageClassification.from_pretrained(
                "microsoft/cvt-13",
                num_labels=self.model_config.n_classes,
                ignore_mismatched_sizes=True
            )
        else:
            deit_config = CvtConfig(
                num_labels=self.model_config.n_classes,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            )
            self.model = CvtForImageClassification(
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
