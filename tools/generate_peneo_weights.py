import argparse
import glob
import json
import os
from shutil import copy

import torch
from transformers import AutoModel, AutoTokenizer, PretrainedConfig

from model.backbone_mapping import BACKBONE_MAPPING


def generate_vie_weights(backbone_name_or_path: str):
    """Generate PEneo config and weights from a given backbone model

    Parameters
    ----------
    backbone_name_or_path : str
        Name or path of the backbone model

    """
    backbone_config = None
    backbone_name = None
    backbone_base_prefix = None
    backbone_preprocessor_config = None
    for vbn in BACKBONE_MAPPING.keys():
        if vbn in backbone_name_or_path:
            backbone_config = PretrainedConfig.from_pretrained(backbone_name_or_path)
            backbone_model_path = os.path.join(
                backbone_name_or_path, "pytorch_model.bin"
            )
            backbone_name = vbn
            backbone_base_prefix = BACKBONE_MAPPING[vbn].model.base_model_prefix

            backbone_tokenizer = AutoTokenizer.from_pretrained(backbone_name_or_path)

            backbone_image_preprocessor = BACKBONE_MAPPING[vbn].image_processor
            if backbone_image_preprocessor is not None:
                local_backbone_preprocessor_config_path = os.path.join(
                    backbone_name_or_path, "processor_config.json"
                )
                if os.path.exists(local_backbone_preprocessor_config_path):
                    backbone_preprocessor_config, _ = (
                        backbone_image_preprocessor.get_image_processor_dict(
                            local_backbone_preprocessor_config_path
                        )
                    )
                else:
                    backbone_preprocessor_config, _ = (
                        backbone_image_preprocessor.get_image_processor_dict(
                            backbone_name_or_path
                        )
                    )
                backbone_preprocessor_config["apply_ocr"] = False
            break

    if backbone_config is None:
        raise ValueError(
            f"Backbone name or path {backbone_name_or_path} is not supported, "
            f"please use one of {list(BACKBONE_MAPPING.keys())}"
        )

    config = {
        "model_type": "peneo",
        "peneo_decoder_shrink": True,
        "peneo_classifier_num_layers": 2,
        "peneo_loss_ratio": [1.0, 1.0, 1.0, 1.0, 1.0],
        "peneo_category_weights": [1, 10, 10],
        "peneo_ohem_num_positive": -1,
        "peneo_ohem_num_negative": -1,
        "peneo_downstream_speedup_ratio": 30.0,
        "backbone_name": backbone_name,
        "backbone_config": backbone_config.to_dict(),
    }

    try:
        if os.path.exists(backbone_name_or_path):
            # load local model weights
            backbone_model_weights = torch.load(backbone_model_path, map_location="cpu")
        else:
            # load from huggingface
            backbone_model = AutoModel.from_pretrained(backbone_name_or_path)
            backbone_model_weights = backbone_model.state_dict()
        new_backbone_model_weights = {}
        for k, v in backbone_model_weights.items():
            new_backbone_model_weights[
                f"backbone.{k.replace(f'{backbone_base_prefix}.', '')}"
            ] = v
    except Exception as e:
        new_backbone_model_weights = None
        print(f"Failed to load backbone model weights: {e}")

    return (
        config,
        new_backbone_model_weights,
        backbone_tokenizer,
        backbone_preprocessor_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone_name_or_path",
        type=str,
        help="Name or path of the backbone model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory to save the config",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if os.path.exists(args.backbone_name_or_path):
        # local model, copy config, tokenizer utils to the output directory
        for file in glob.glob(args.backbone_name_or_path + "/*"):
            if os.path.basename(file) in [
                "config.json",
                "pytorch_model.bin",
                "model.safetensors",
                "tf_model.h5",
            ]:
                continue

            src = file
            dst = os.path.join(args.output_dir, os.path.basename(file))
            copy(src, dst)

    config, weights, tokenizer, preprocessor_config = generate_vie_weights(
        args.backbone_name_or_path
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_dir)

    if weights is not None:
        torch.save(weights, os.path.join(args.output_dir, "pytorch_model.bin"))

    if preprocessor_config is not None:
        with open(os.path.join(args.output_dir, "preprocessor_config.json"), "w") as f:
            json.dump(preprocessor_config, f, indent=4)
