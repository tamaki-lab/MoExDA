import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List
from transformers import ViTForImageClassification, ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTModel,
    ViTLayer,
    ViTEncoder,
    ViTEmbeddings,
    ViTAttention,
    ViTSdpaAttention,
    ViTIntermediate,
    ViTOutput,
    ViTPooler,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput, ImageClassifierOutput
from model import ModelConfig, ModelOutput
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CustomModelOutput:
    logits: List[torch.Tensor]
    loss: Optional[List[torch.Tensor]] = None
    cls_tokens: Optional[List[torch.Tensor]] = None


class Normalization(nn.Module):
    def __init__(self, input_size, return_stats, affine, eps):
        super().__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine
        self.dim = None

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=self.dim, keepdim=True)
        std = (x.var(dim=self.dim, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std


class PONO(Normalization):
    def __init__(self, input_size=None, return_stats=False, affine=False, eps=1e-5):
        super().__init__(input_size, return_stats, affine, eps)
        self.dim = 2  # patch dimension for ViT


class IN(Normalization):
    def __init__(self, input_size=None, return_stats=False, affine=False, eps=1e-5):
        super().__init__(input_size, return_stats, affine, eps)
        self.dim = 1  # channel dimension for ViT


VIT_ATTENTION_CLASSES = {
    "eager": ViTAttention,
    "sdpa": ViTSdpaAttention,
}


class MyViTOutput(ViTOutput):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class MoExDA(nn.Module):
    def __init__(self, moexda_dict: dict) -> None:
        super().__init__()
        # moexda_dict['moment_type']  # PONO or IN (nn.Module)
        self.norm = PONO() if moexda_dict['moment_type'] == 'pono' else IN()
        self.exchange_direction = moexda_dict['exchange_direction']  # 'edge_to_rgb', 'rgb_to_edge', or 'bidirectional'
        self.stop_gradient = moexda_dict['stop_gradient']  # bool, whether to stop gradient flow

    def exchange(self, x, src_mean, src_std, tgt_mean, tgt_std):
        if self.stop_gradient:
            tgt_mean = tgt_mean.detach()
            tgt_std = tgt_std.detach()
        return ((x - src_mean) / src_std) * tgt_std + tgt_mean

    def forward(self, edge_feat: torch.Tensor, rgb_feat: torch.Tensor):
        # Get only stats; discard normalized output
        _, edge_mean, edge_std = self.norm(edge_feat)
        _, rgb_mean, rgb_std = self.norm(rgb_feat)

        if self.exchange_direction == 'edge_to_rgb':
            edge_feat = self.exchange(edge_feat, edge_mean, edge_std, rgb_mean, rgb_std)
        elif self.exchange_direction == 'rgb_to_edge':
            rgb_feat = self.exchange(rgb_feat, rgb_mean, rgb_std, edge_mean, edge_std)
        elif self.exchange_direction == 'bidirectional':
            edge_feat_new = self.exchange(edge_feat, edge_mean, edge_std, rgb_mean, rgb_std)
            rgb_feat_new = self.exchange(rgb_feat, rgb_mean, rgb_std, edge_mean, edge_std)
            edge_feat, rgb_feat = edge_feat_new, rgb_feat_new
        else:
            raise ValueError(f"Invalid exchange_direction: {self.exchange_direction}")

        return edge_feat, rgb_feat


class MoExViTLayer(ViTLayer):
    def __init__(self, model_config: ViTConfig, moexda_dict: Optional[dict]) -> None:
        super().__init__(model_config)
        self.edge_attention = VIT_ATTENTION_CLASSES[model_config._attn_implementation](model_config)
        self.edge_intermediate = ViTIntermediate(model_config)
        self.edge_output = ViTOutput(model_config)
        self.edge_layernorm_before = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.edge_layernorm_after = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.moexda_dict = moexda_dict
        self.position = moexda_dict['exchange_position'] if moexda_dict else None
        self.moexda = MoExDA(moexda_dict) if moexda_dict else None

    def forward(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rgb_hidden_states: Optional[torch.Tensor] = None,
        use_moex: bool = False,  # whether to use MoEx
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if rgb_hidden_states is not None and hidden_states is not None:

            if use_moex and self.moexda_dict["exchange_position"] == "BeforeMHA":  # type: ignore
                # Apply MoExDA before self-attention
                hidden_states, rgb_hidden_states = self.moexda(rgb_hidden_states, hidden_states)  # type: ignore

            self_rgb_attention_outputs = self.attention(
                self.layernorm_before(rgb_hidden_states),  # in ViT, layernorm is applied before self-attention
                head_mask,
                output_attentions=output_attentions,
            )

            self_attention_outputs = self.edge_attention(
                self.edge_layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
                head_mask,
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            rgb_attention_output = self_rgb_attention_outputs[0]
            rgb_outputs = self_rgb_attention_outputs[1:]  # add self attentions if we output attention weights
            # first residual connection
            hidden_states = attention_output + hidden_states
            rgb_hidden_states = rgb_attention_output + rgb_hidden_states

            if use_moex and self.moexda_dict["exchange_position"] == "AfterMHA":  # type: ignore
                # Apply MoExDA after self-attention
                hidden_states, rgb_hidden_states = self.moexda(rgb_hidden_states, hidden_states)  # type: ignore

            # in ViT, layernorm is also applied after self-attention
            layer_output = self.edge_layernorm_after(hidden_states)
            rgb_layer_output = self.layernorm_after(rgb_hidden_states)

            if use_moex and self.moexda_dict["exchange_position"] == "BeforeMLP":  # type: ignore
                # Apply MoExDA before MLP
                layer_output, rgb_layer_output = self.moexda(rgb_layer_output, layer_output)  # type: ignore

            layer_output = self.edge_intermediate(layer_output)
            rgb_layer_output = self.intermediate(rgb_layer_output)

            # second residual connection is done here
            layer_output = self.edge_output(layer_output, hidden_states)
            rgb_layer_output = self.output(rgb_layer_output, rgb_hidden_states)

            if use_moex and self.moexda_dict["exchange_position"] == "AfterMLP":  # type: ignore
                # Apply MoExDA after MLP
                layer_output, rgb_layer_output = self.moexda(rgb_layer_output, layer_output)  # type: ignore

            outputs = (layer_output,) + outputs
            rgb_outputs = (rgb_layer_output,) + rgb_outputs

        elif hidden_states is not None and rgb_hidden_states is None:
            rgb_layer_output = None
            rgb_outputs = None
            self_attention_outputs = self.edge_attention(
                self.edge_layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
                head_mask,
                output_attentions=output_attentions,
            )
            attention_output = self_attention_outputs[0]
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

            # first residual connection
            hidden_states = attention_output + hidden_states

            # in ViT, layernorm is also applied after self-attention
            layer_output = self.edge_layernorm_after(hidden_states)
            layer_output = self.edge_intermediate(layer_output)

            # second residual connection is done here
            layer_output = self.edge_output(layer_output, hidden_states)

            outputs = (layer_output,) + outputs

        return [outputs, rgb_outputs]  # type: ignore


class MoExLayerViTEncoder(ViTEncoder):
    def __init__(
            self,
            config: ViTConfig,
            moex_layers: Optional[list] = None,
            cos_sim: bool = False,
            moexda_dict: Optional[dict] = None,
    ) -> None:
        super().__init__(config)
        self.moex_layers = moex_layers
        self.cos_sim = cos_sim
        self.moexda_dict = moexda_dict
        self.layer = nn.ModuleList(
            [MoExViTLayer(config, self.moexda_dict) for _ in range(config.num_hidden_layers)]
        )
        self.all_edge_features = [] if self.cos_sim else None
        self.all_rgb_features = [] if self.cos_sim else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        rgb_hidden_states: Optional[torch.Tensor] = None,
    ) -> Union[tuple, BaseModelOutput]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_rgb_hidden_states = () if output_hidden_states else None
        all_self_rgb_attentions = () if output_attentions else None
        use_moex = False
        for i, layer_module in enumerate(self.layer):
            if self.moex_layers is not None:
                use_moex = i in self.moex_layers
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs_edge_rgb_list = layer_module(
                hidden_states=hidden_states,
                head_mask=layer_head_mask,
                output_attentions=output_attentions,
                rgb_hidden_states=rgb_hidden_states,
                use_moex=use_moex
            )

            # Update current_hidden_states to the output of this layer
            if isinstance(layer_outputs_edge_rgb_list, list):
                layer_outputs = layer_outputs_edge_rgb_list[0]
                hidden_states = layer_outputs[0]
                rgb_layer_outputs = layer_outputs_edge_rgb_list[1]
                if rgb_layer_outputs is not None:
                    rgb_hidden_states = rgb_layer_outputs[0]
                else:
                    rgb_hidden_states = None

                # Collect attention outputs if needed
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                all_self_rgb_attentions = all_self_rgb_attentions + (rgb_layer_outputs[1],)

            # Collect features for cosine similarity
            if self.cos_sim:
                self.all_edge_features.append(hidden_states)
                self.all_rgb_features.append(rgb_hidden_states)

        # Collect final hidden states if needed
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            all_rgb_hidden_states = all_rgb_hidden_states + (rgb_hidden_states,)

        # Prepare the return structure based on the 'return_dict' flag
        if not return_dict:
            tuple(
                v for v in [
                    hidden_states,
                    rgb_hidden_states,
                    all_hidden_states,
                    all_rgb_hidden_states,
                    all_self_rgb_attentions,
                    all_self_rgb_attentions] if v is not None)

        last_hidden_states = [hidden_states, rgb_hidden_states]
        all_states = [all_hidden_states, all_rgb_hidden_states]
        attentions = [all_self_attentions, all_self_rgb_attentions]

        return BaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=all_states,
            attentions=attentions,
        )


class MoExLayerViTModel(ViTModel):
    def __init__(
            self,
            model_config: ViTConfig,
            moex_layers: Optional[list] = None,
            cos_sim: bool = False,
            add_pooling_layer: bool = True,
            use_mask_token: bool = False,
            moexda_dict: Optional[dict] = None,
    ) -> None:
        super().__init__(model_config)
        self.encoder = MoExLayerViTEncoder(
            model_config,
            moex_layers=moex_layers,
            cos_sim=cos_sim,
            moexda_dict=moexda_dict)
        self.edge_embeddings = ViTEmbeddings(model_config, use_mask_token=False)
        self.edge_layernorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.pooler = ViTPooler(model_config) if add_pooling_layer else None
        self.edge_pooler = ViTPooler(model_config) if add_pooling_layer else None

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rgb_values: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # if rgb_values is None:
        #     raise ValueError("You have to specify rgb_values")
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)
        if rgb_values is not None:
            rgb_embedding_output = self.embeddings(
                rgb_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
            )
        else:
            rgb_embedding_output = None
        embedding_output = self.edge_embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            head_mask=head_mask,
            hidden_states=embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            rgb_hidden_states=rgb_embedding_output,
        )
        sequence_output = encoder_outputs[0][0]
        rgb_sequence_output = encoder_outputs[0][1]
        sequence_output = self.edge_layernorm(sequence_output)
        if rgb_sequence_output is not None:
            rgb_sequence_output = self.layernorm(rgb_sequence_output)
        pooled_output = self.edge_pooler(sequence_output) if self.edge_pooler is not None else None
        rgb_pooled_output = self.pooler(rgb_sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            rgb_head_outputs = (rgb_sequence_output, rgb_pooled_output) if rgb_pooled_output is not None else (
                rgb_sequence_output,
            )
            return head_outputs + encoder_outputs[1][0][1:], rgb_head_outputs + encoder_outputs[1][1][1:]

        last_hidden_state = [sequence_output, rgb_sequence_output]
        pooler_output = [pooled_output, rgb_pooled_output]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states[0],
            attentions=encoder_outputs.attentions[0],
        )


class MoExLayerViTForImageClassification(ViTForImageClassification):
    def __init__(
            self,
            model_config: ViTConfig,
            moex_layers: Optional[list] = None,
            cos_sim: bool = False,
            moexda_dict: Optional[dict] = None,):

        super().__init__(model_config)
        self.vit = MoExLayerViTModel(
            model_config,
            add_pooling_layer=False,
            moex_layers=moex_layers,
            cos_sim=cos_sim,
            moexda_dict=moexda_dict,
        )
        self.edge_classifier = nn.Linear(model_config.hidden_size,
                                         model_config.num_labels) if model_config.num_labels > 0 else nn.Identity()
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        rgb_values: Optional[torch.Tensor] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values=pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
            rgb_values=rgb_values,
        )

        sequence_output = outputs[0][0]
        rgb_sequence_output = outputs[0][1]

        logits = self.edge_classifier(sequence_output[:, 0, :])
        if rgb_sequence_output is not None:
            rgb_logits = self.classifier(rgb_sequence_output[:, 0, :])
        else:
            rgb_logits = None
        loss = None
        rgb_loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=[loss, rgb_loss],
            logits=[logits, rgb_logits],
            hidden_states=[sequence_output, rgb_sequence_output],
            attentions=outputs.attentions,
        )


class MoExLayerViT(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config
        self.prepare_model(checkpoint_path=model_config.checkpoint_path)
        self.moex_layers = model_config.moex_layers
        self.cos_sim = model_config.cos_sim
        self.moexda_dict = model_config.moexda_dict

    def prepare_model(self, checkpoint_path=None):
        self.model = MoExLayerViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=self.model_config.n_classes,
            ignore_mismatched_sizes=True,
            moex_layers=self.model_config.moex_layers,
            cos_sim=self.model_config.cos_sim,
            moexda_dict=self.model_config.moexda_dict,
        )
        if self.model_config.moex_layers is not None and not all(0 <= x <= 11 for x in self.model_config.moex_layers):
            raise ValueError("Moex layers must be between 0 and 11")

        updated_model_state_dict = self.separate_load_from_state_dict(self.model.state_dict())
        self.model.load_state_dict(updated_model_state_dict)

        if checkpoint_path is not None:
            state_dict = self.load_state_from_ckpt(checkpoint_path)
            self.model.load_state_dict(state_dict)

    def separate_load_from_state_dict(self, state_dict):
        edge_state_dict = OrderedDict()
        rgb_state_dict = OrderedDict()
        updated_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if "edge" in key:
                # edgeが含まれている場合、edge_state_dict に key と value を追加
                edge_state_dict[key] = value
            else:
                # 含まれていない場合、rgb_state_dict に key と value を追加
                rgb_state_dict[key] = value

        for edge_key in list(edge_state_dict.keys()):
            key = edge_key.replace("edge_", "")
            if key in rgb_state_dict:
                edge_state_dict[edge_key] = rgb_state_dict[key]
            else:
                raise ValueError("The keys of both dictionaries do not match.")
        updated_state_dict.update(rgb_state_dict)
        updated_state_dict.update(edge_state_dict)
        return updated_state_dict

    def load_state_from_ckpt(self, checkpoint_path):
        # .ckptファイルから重みをロード
        checkpoint = torch.load(checkpoint_path)
        state_dict = {
            key.replace("model.model.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        classifier_keys = [key for key in state_dict.keys() if "classifier" in key]
        for classifier in classifier_keys:
            state_dict[classifier] = self.model.state_dict()[classifier]
        # state_dcitの整合性の確認
        if not set(self.model.state_dict().keys()) == set(state_dict.keys()):
            raise ValueError("The keys of both dictionaries do not match.")
        return state_dict

    def forward(
        self,
        pixel_values: torch.Tensor,
        rgb_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CustomModelOutput:

        B, T, C, H, W = pixel_values.shape  # sobel_values : (B,T,C,H,W)
        pixel_values = pixel_values.reshape(-1, C, H, W)  # (B,T,C,H,W) -> (B*T,C,H,W)
        if rgb_values is not None:
            rgb_values = rgb_values.reshape(-1, C, H, W)  # (B,T,C,H,W) -> (B*T,C,H,W)

        logits = 0
        num_labels = self.model_config.n_classes

        output = self.model(pixel_values=pixel_values, rgb_values=rgb_values)
        logits += output.logits[0]
        rgb_logits = output.logits[1]

        logits = logits.view(B, T, -1)
        logits = logits.mean(dim=1)
        if rgb_logits is not None:
            rgb_logits = rgb_logits.view(B, T, -1)
            rgb_logits = rgb_logits.mean(dim=1)
        else:
            rgb_logits = None

        loss_fct = CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss = None

        if rgb_logits is not None and labels is not None:
            rgb_loss = loss_fct(rgb_logits.view(-1, num_labels), labels.view(-1))
        else:
            rgb_loss = None

        sequence_output = output.hidden_states[0]
        rgb_sequence_output = output.hidden_states[1]

        cls_token = sequence_output[:, 0, :]
        if rgb_sequence_output is not None:
            rgb_cls_token = rgb_sequence_output[:, 0, :]
        else:
            rgb_cls_token = None

        return CustomModelOutput(
            logits=[logits, rgb_logits],
            loss=[loss, rgb_loss],
            cls_tokens=[cls_token, rgb_cls_token],
        )
