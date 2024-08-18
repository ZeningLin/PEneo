import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch
import tqdm
from PIL import Image
from transformers import HfArgumentParser, PreTrainedTokenizer, ProcessorMixin, set_seed

from data.data_utils import normalize_bbox, sort_boxes, string_f2h
from model import PEneoConfig, PEneoModel
from model.backbone_mapping import BACKBONE_MAPPING
from model.peneo_decoder import HandshakingTaggingScheme
from pipeline.decode import sample_decode_peneo

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
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


@dataclass
class DataArguments:
    dir_image: str = field(
        default=None,
        metadata={"help": "Directory of the sample image"},
    )

    dir_ocr: str = field(
        default=None,
        metadata={"help": "Directory of the OCR result"},
    )

    apply_ocr: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply OCR to the image or not, by default False. "
            "If true, will use the built-in Tesseract OCR engine of transformers processor",
        },
    )


class InferenceService:
    def __init__(self, model_args: ModelArguments, data_args: DataArguments):
        logger.info(f"Loading model from {model_args.model_name_or_path}")
        self.config: PEneoConfig = PEneoConfig.from_pretrained(
            (
                model_args.config_name
                if model_args.config_name
                else model_args.model_name_or_path
            ),
        )
        self.config.inference_mode = True
        self.backbone_info = BACKBONE_MAPPING.get(self.config.backbone_name, None)

        if self.backbone_info is None:
            logger.error(
                f"Invalid backbone name {self.config.backbone_name}",
                f"Available backbones are {list(BACKBONE_MAPPING.keys())}",
            )
            raise ValueError()

        self.apply_ocr = data_args.apply_ocr
        if self.apply_ocr:
            logger.info(
                "apply_ocr is set to True, using built-in Tesseract OCR engine of transformers processor. The provided OCR path will be ignored.",
            )
        self.processor = self.backbone_info.processor.from_pretrained(
            model_args.model_name_or_path, use_fast=True, apply_ocr=data_args.apply_ocr
        )
        if isinstance(self.processor, ProcessorMixin):
            self.tokenizer: PreTrainedTokenizer = self.processor.tokenizer
            self.image_processor = self.processor.image_processor
            self.require_image = True
        else:
            self.tokenizer: PreTrainedTokenizer = self.processor
            self.image_processor = None
            self.require_image = False
        self.tokenizer_fetcher = self.backbone_info.tokenizer_fetcher

        self.max_token_len = self.backbone_info.max_token_len
        self.add_cls_token = self.backbone_info.add_cls_token
        self.add_sep_token = self.backbone_info.add_sep_token
        self.padding_side = self.tokenizer.padding_side

        self.model = PEneoModel.from_pretrained(
            model_args.model_name_or_path,
            config=self.config,
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.eval()

        self.handshaking_tagger = HandshakingTaggingScheme()
        logger.info(f"Model loaded from {model_args.model_name_or_path}")

        self.NO_BATCH_KEYS = []
        self.NO_TENSOR_KEYS = [
            "text",
            "relations",
            "line_extraction_shaking_tag",
            "ent_linking_head_rel_shaking_tag",
            "ent_linking_tail_rel_shaking_tag",
            "line_grouping_head_rel_shaking_tag",
            "line_grouping_tail_rel_shaking_tag",
        ]

    def _special_text_replace(self, line_text: str) -> str:
        line_text = line_text.replace("☐", "")
        line_text = line_text.replace("☑", "")
        line_text = line_text.replace("\uf702", "")
        line_text = line_text.replace("\uf703", "")
        line_text = line_text.replace("Tοpic", "Topic")  # ? Magic, don't remove
        line_text = line_text.replace("á", "a")
        line_text = line_text.replace("é", "e")
        line_text = line_text.replace("í", "i")
        line_text = line_text.replace("ó", "o")
        line_text = line_text.replace("ú", "u")
        line_text = line_text.replace("ü", "u")
        line_text = line_text.replace("–", "-")
        line_text = line_text.replace("‘", "'")
        line_text = line_text.replace("’", "'")
        line_text = line_text.replace("“", '"')
        line_text = line_text.replace("—", "-")
        line_text = line_text.replace("™", "TM")
        line_text = line_text.replace("§", "")
        line_text = line_text.replace("¢", "")

        return string_f2h(line_text)

    def preprocess(
        self,
        image_path: Union[str, List[str]],
        ocr_path: Union[str, List[str]] = None,
    ):
        if isinstance(image_path, str):
            if os.path.isdir(image_path):
                image_path_list = os.listdir(image_path)
                image_path_list = [os.path.join(image_path, x) for x in image_path_list]
            else:
                image_path_list = [image_path]
        else:
            raise ValueError("image_path must be a string")

        if not ocr_path:
            assert self.apply_ocr, "OCR path must be provided if apply_ocr is False"
            ocr_path_list = [None] * len(image_path_list)
        else:
            if isinstance(ocr_path, str):
                if os.path.isdir(ocr_path):
                    ocr_path_list = os.listdir(ocr_path)
                    ocr_path_list = [os.path.join(ocr_path, x) for x in ocr_path_list]
                else:
                    ocr_path_list = [ocr_path]
            else:
                raise ValueError("ocr_path must be a string")

        assert len(image_path_list) == len(
            ocr_path_list
        ), "Number of image and OCR paths must be the same"

        for i, curr_image_path in enumerate(image_path_list):
            image = Image.open(curr_image_path).convert("RGB")
            image_w, image_h = image.size
            if self.require_image or self.apply_ocr:
                image_features = self.image_processor(images=image, return_tensors="pt")
                image_return = image_features["pixel_values"].to(self.device)
            else:
                image_return = None

            curr_ocr_path = ocr_path_list[i]
            if self.apply_ocr or not curr_ocr_path or not os.path.exists(curr_ocr_path):
                line_text_list = image_features["words"][0]
                line_box_list = image_features["boxes"][0]
            else:
                curr_ocr = json.load(open(curr_ocr_path, "r", encoding="utf-8"))
                line_text_list, line_box_list = [], []
                for line_info in curr_ocr:
                    line_text_list.append(line_info["text"])
                    if "bbox" in line_info:
                        line_box_list.append(line_info["bbox"])
                    else:
                        line_box_list.append(line_info["box"])

            ro_sorted_box_idx_list = sort_boxes(line_box_list)

            texts = []
            input_ids = []
            bbox = []
            orig_bbox = []
            curr_token_idx = 0
            for ro_sorted_idx in ro_sorted_box_idx_list:
                line_text = line_text_list[ro_sorted_idx]
                line_text = self._special_text_replace(line_text)
                line_tokens = self.tokenizer.tokenize(line_text)
                line_token_ids = self.tokenizer.convert_tokens_to_ids(line_tokens)
                if len(line_tokens) == 0:
                    continue
                line_token_len = len(line_token_ids)
                if curr_token_idx + line_token_len > self.max_token_len:
                    # reach max token length, break
                    break

                if self.tokenizer_fetcher is not None:
                    line_sos_processed_tokens = self.tokenizer_fetcher(
                        line_text, line_tokens
                    )
                else:
                    line_sos_processed_tokens = line_tokens

                line_orig_bbox = line_box_list[ro_sorted_idx]
                line_norm_bbox = normalize_bbox(line_orig_bbox, (image_w, image_h))

                orig_bbox.extend([line_orig_bbox] * line_token_len)
                bbox.extend([line_norm_bbox] * line_token_len)
                texts.extend(line_sos_processed_tokens)
                input_ids.extend(line_token_ids)

            if self.add_cls_token:
                input_ids = [self.tokenizer.cls_token_id] + input_ids
                orig_bbox = [[0, 0, 0, 0]] + orig_bbox
                bbox = [[0, 0, 0, 0]] + bbox
            if self.add_sep_token:
                input_ids.append(self.tokenizer.sep_token_id)
                orig_bbox.append([0, 0, 0, 0])
                bbox.append([0, 0, 0, 0])

            features = {
                "fname": [curr_image_path],
                "image_path": [curr_image_path],
                "input_ids": [input_ids],
                "bbox": [bbox],
                "orig_bbox": [orig_bbox],
                "text": [texts],
            }
            batch = self.tokenizer.pad(
                features,
                padding="longest",
                max_length=512,
                pad_to_multiple_of=8,
                return_tensors=None,
            )
            sequence_length = torch.tensor(batch["input_ids"]).shape[1]
            if self.padding_side == "right":
                batch["bbox"] = [
                    bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox))
                    for bbox in batch["bbox"]
                ]
                batch["orig_bbox"] = [
                    bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox))
                    for bbox in batch["orig_bbox"]
                ]
            else:
                batch["bbox"] = [
                    [[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox
                    for bbox in batch["bbox"]
                ]
                batch["orig_bbox"] = [
                    [[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox
                    for bbox in batch["orig_bbox"]
                ]

            for k, v in batch.items():
                if isinstance(v[0], list):
                    if k not in self.NO_BATCH_KEYS and k not in self.NO_TENSOR_KEYS:
                        batch[k] = torch.tensor(v, dtype=torch.int64).to(self.device)
                    elif k not in self.NO_TENSOR_KEYS:
                        batch[k] = [
                            torch.tensor(vv, dtype=torch.int64).to(self.device)
                            for vv in v
                        ]
                    else:
                        batch[k] = v
                elif isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device)
                else:
                    batch[k] = v

            if image_return is not None:
                batch["image"] = image_return

            yield batch

    @torch.no_grad()
    def inference(self, inputs):
        return self.model(**inputs)

    def postprocess(self, **kwargs):
        return sample_decode_peneo(
            handshaking_tagger=self.handshaking_tagger,
            decode_gt=False,
            **kwargs,
        )[
            0
        ]  # only return parsed kv-pairs

    def run(self, data_args: DataArguments):
        data_getter = self.preprocess(data_args.dir_image, data_args.dir_ocr)
        predictions = []
        inference_start = time.time()
        sample_cnt = 0
        for inputs in tqdm.tqdm(data_getter):
            (
                line_extraction_shaking_outputs,
                ent_linking_h2h_shaking_outputs,
                ent_linking_t2t_shaking_outputs,
                line_grouping_h2h_shaking_outputs,
                line_grouping_t2t_shaking_outputs,
                orig_bboxes,
            ) = self.inference(inputs)
            texts = inputs.get("text")

            for (
                line_extraction_shaking_output,
                ent_linking_h2h_shaking_output,
                ent_linking_t2t_shaking_output,
                line_grouping_h2h_shaking_output,
                line_grouping_t2t_shaking_output,
                orig_bbox,
                text,
            ) in zip(
                line_extraction_shaking_outputs,
                ent_linking_h2h_shaking_outputs,
                ent_linking_t2t_shaking_outputs,
                line_grouping_h2h_shaking_outputs,
                line_grouping_t2t_shaking_outputs,
                orig_bboxes,
                texts,
            ):
                if len(texts) == 0:
                    continue
                seq_len = len(orig_bbox)
                shaking_ind2matrix_ind = [
                    (ind, end_ind)
                    for ind in range(seq_len)
                    for end_ind in list(range(seq_len))[ind:]
                ]
                predictions.append(
                    self.postprocess(
                        line_extraction_shaking=line_extraction_shaking_output,
                        ent_linking_h2h_shaking=ent_linking_h2h_shaking_output,
                        ent_linking_t2t_shaking=ent_linking_t2t_shaking_output,
                        line_grouping_h2h_shaking=line_grouping_h2h_shaking_output,
                        line_grouping_t2t_shaking=line_grouping_t2t_shaking_output,
                        text=text,
                        shaking_ind2matrix_ind=shaking_ind2matrix_ind,
                    )
                )
                sample_cnt += 1

        inference_end = time.time()
        avg_inference_time = (inference_end - inference_start) / sample_cnt
        logger.info(f"Inference time: {inference_end - inference_start:.2f}s")
        logger.info(f"Average inference time: {avg_inference_time:.2f}s")

        return predictions


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(42)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )

    inference_service = InferenceService(model_args, data_args)
    predictions = inference_service.run(data_args)

    logger.info("Predictions:")
    for pred in predictions:
        logger.info(pred)


if __name__ == "__main__":
    main()
