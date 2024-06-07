import logging
from dataclasses import dataclass, field
from typing import Optional

import torch.onnx
from transformers import HfArgumentParser, ProcessorMixin

from model import PEneoConfig, PEneoModel
from model.backbone_mapping import BACKBONE_MAPPING

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    output_path: str = field(metadata={"help": "Path to save the exported onnx model"})
    backbone_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models",
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )


def main():
    parser = HfArgumentParser(ModelArguments)
    model_args = parser.parse_args_into_dataclasses()[0]

    # Load Model
    config = PEneoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
    )
    config.export_onnx = True
    backbone_info = BACKBONE_MAPPING.get(config.backbone_name, None)
    if backbone_info is None:
        logger.error(
            f"Invalid backbone name {config.backbone_name}",
            f"Available backbones are {list(BACKBONE_MAPPING.keys())}",
        )
        raise ValueError()

    if "layoutlmv3-base-chinese" in config.backbone_name:
        tokenizer_use_fast = False
    else:
        tokenizer_use_fast = True
    processor = backbone_info.processor.from_pretrained(
        model_args.model_name_or_path, use_fast=tokenizer_use_fast
    )
    if isinstance(processor, ProcessorMixin):
        tokenizer = processor.tokenizer
        image_processor = processor.image_processor
    else:
        tokenizer = processor
        image_processor = None

    model = PEneoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )

    # backbone args
    input_ids = torch.randint(1, 200000, (1, 512))
    bbox = torch.zeros((1, 512, 4)).long()
    attention_mask = torch.ones((1, 512)).long()

    # peneo decoder args
    orig_bbox = torch.zeros((1, 512, 4)).long()

    input_dict = {
        "input_ids": input_ids,
        "bbox": bbox,
        "orig_bbox": orig_bbox,
        "attention_mask": attention_mask,
    }
    output_names = [
        "line_extraction_shaking_outputs",
        "ent_linking_h2h_shaking_outputs",
        "ent_linking_t2t_shaking_outputs",
        "line_grouping_h2h_shaking_outputs",
        "line_grouping_t2t_shaking_outputs",
        "return_orig_bbox",
    ]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence"},
        "bbox": {0: "batch_size", 1: "sequence"},
        "orig_bbox": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},
    }

    # image_input
    if backbone_info.has_visual_embeds:
        input_image = torch.randint(0, 255, (3, 1000, 1000))
        input_image = image_processor(input_image, return_tensors="pt")["pixel_values"]
        input_dict.update({"image": input_image})
        dynamic_axes.update({"image": {0: "batch_size"}})
    input_keys = list(input_dict.keys())
    input_values = tuple(input_dict.values())

    print(input_keys)
    print(output_names)

    torch.onnx.export(
        model,
        input_values,
        model_args.output_path,
        input_names=input_keys,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )


if __name__ == "__main__":
    main()
