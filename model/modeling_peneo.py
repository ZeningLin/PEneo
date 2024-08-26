import inspect
import logging
import math

import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .backbone_mapping import BACKBONE_MAPPING
from .configuration_peneo import PEneoConfig
from .peneo_decoder import PEneoDecoder

logger = logging.getLogger(__name__)


class PEneoPreTrainedModel(PreTrainedModel):
    config_class = PEneoConfig
    base_model_prefix = "backbone"

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.data.normal_(
                    mean=0.0,
                    std=(
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.num_hidden_layers)
                    ),
                )


class PEneoModel(PEneoPreTrainedModel):
    """
    Visual information extraction model class,
    with switchable backbones and downstream heads.
    """

    def __init__(self, config: PEneoConfig, backbone_name_or_path: str = None) -> None:
        super().__init__(config)
        self.backbone_name = config.backbone_name
        self.backbone_info = BACKBONE_MAPPING[config.backbone_name]
        if config.backbone_config is None and backbone_name_or_path is None:
            raise ValueError(
                "You cannot initialize a model with a config file that has no backbone config "
                "and without specifying the path to a pretrained model."
            )

        # backbone initialization
        if backbone_name_or_path is not None:
            # if backbone_name_or_path is not None (Use in the case of finetuning)
            # load from huggingface_hub or local path
            if backbone_name_or_path == "auto":
                self.backbone = self.backbone_info.model.from_pretrained(
                    self.backbone_info.hf_name
                )
            else:
                try:
                    self.backbone = self.backbone_info.model.from_pretrained(
                        backbone_name_or_path
                    )
                except OSError:
                    logger.warning(
                        f"Could not load pretrained model from {backbone_name_or_path}"
                        f"Load from {self.backbone_info.hf_name} in huggingface_hub instead."
                    )
                    self.backbone = self.backbone_info.model.from_pretrained(
                        self.backbone_info.hf_name
                    )
            if config.backbone_config is None:
                config.backbone_config = self.backbone.config.to_dict()
        else:
            # if backbone_name_or_path is None (Use in the case of eval&predict)
            # initialize with config, then load from saved finetuned weights
            self.backbone = self.backbone_info.model(
                self.backbone_info.config.from_dict(config.backbone_config)
            )

        self.dropout = nn.Dropout(config.backbone_config["hidden_dropout_prob"])

        self.loss_ratio = config.peneo_loss_ratio
        if self.loss_ratio is not None:
            assert len(self.loss_ratio) == 5, "loss_ratio must be a list of 5 elements"

        if "lilt" in self.backbone_name.lower():
            downstream_input_size = (
                config.backbone_config["hidden_size"]
                + config.backbone_config["hidden_size"]
                // config.backbone_config["channel_shrink_ratio"]
            )
        else:
            downstream_input_size = config.backbone_config["hidden_size"]
        self.peneo_decoder = PEneoDecoder(
            config=config, input_size=downstream_input_size
        )

    def _init_weights(self, module) -> None:
        self.backbone._init_weights(module)

    def forward(
        self, input_ids, bbox, orig_bbox, attention_mask, image=None, **kwargs
    ) -> ModelOutput:
        ## ! For onnx
        kwargs.update(
            {
                "input_ids": input_ids,
                "bbox": bbox,
                "orig_bbox": orig_bbox,
                "attention_mask": attention_mask,
                "image": image,
            }
        )
        ## ! onnx

        backbone_kwarg_names = inspect.signature(
            self.backbone.forward
        ).parameters.values()
        backbone_kwarg_names = [bk.name for bk in backbone_kwarg_names]
        backbone_kwargs = {
            bk_name: kwargs.get(bk_name, None) for bk_name in backbone_kwarg_names
        }

        backbone_output = self.backbone(**backbone_kwargs)

        # remove visual embeds, cls tokens according to backbone config
        hidden_states = backbone_output[0]
        bbox = kwargs.pop("bbox", None)
        orig_bbox = kwargs.pop("orig_bbox", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if self.backbone_info.has_visual_embeds:
            seq_len = backbone_kwargs["input_ids"].shape[1]
            if self.backbone_info.add_cls_token:
                # remove visual embeds
                hidden_states = hidden_states[:, 1:seq_len]
                bbox = bbox[:, 1:seq_len] if bbox is not None else None
                orig_bbox = orig_bbox[:, 1:seq_len] if orig_bbox is not None else None
                attention_mask = (
                    attention_mask[:, 1:seq_len] if attention_mask is not None else None
                )
            else:
                hidden_states = hidden_states[:, :seq_len]
                bbox = bbox[:, :seq_len] if bbox is not None else None
                orig_bbox = orig_bbox[:, :seq_len] if orig_bbox is not None else None
                attention_mask = (
                    attention_mask[:, :seq_len] if attention_mask is not None else None
                )
        else:
            if self.backbone_info.add_cls_token:
                # remove CLS token
                hidden_states = hidden_states[:, 1:]
                bbox = bbox[:, 1:] if bbox is not None else None
                orig_bbox = orig_bbox[:, 1:] if orig_bbox is not None else None
                attention_mask = (
                    attention_mask[:, 1:] if attention_mask is not None else None
                )

        hidden_states = self.dropout(hidden_states)

        decoder_output = self.peneo_decoder(
            sequence_output=hidden_states,
            bbox=bbox,
            orig_bbox=orig_bbox,
            attention_mask=attention_mask,
            **kwargs,
        )

        return decoder_output
