from typing import Union

import torch
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from model.peneo_decoder import HandshakingTaggingScheme


class DataCollatorForPEneo:
    """
    Data collator for PEneo model. Used to pad the input sequences,
    generate the handshaking tags, and process the images.

    Parameters
    ----------
    tokenizer : Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
        Tokenizer for the model.
    image_processor : ProcessorMixin, optional
        Image processor for the model, by default None
    padding : str, optional
        Padding type, by default "longest"
    max_length : int, optional
        Maximum length of the returned list and optionally padding length (see above).
    pad_to_multiple_of : int, optional
        If set will pad the sequence to a multiple of the provided value.
        This is especially useful to enable the use of Tensor Cores on
        NVIDIA hardware with compute capability >=7.5 (Volta).
    label_pad_token_id : int, optional
        The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    require_image : bool, optional
        Whether the backbone model requires an image input, by default True
    add_cls_token : bool, optional
        Whether the backbone model requires a cls token, by default True
    add_sep_token : bool, optional
        Whether the backbone model requires a sep token, by default True

    """

    PADDING_TYPE = ["longest", "max_length"]
    NO_BATCH_KEYS = []
    NO_TENSOR_KEYS = [
        "text",
        "relations",
        "line_extraction_shaking_tag",
        "ent_linking_head_rel_shaking_tag",
        "ent_linking_tail_rel_shaking_tag",
        "line_grouping_head_rel_shaking_tag",
        "line_grouping_tail_rel_shaking_tag",
    ]

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        image_processor: ProcessorMixin = None,
        padding: str = "longest",
        max_length: int = 512,
        pad_to_multiple_of: int = 8,
        label_pad_token_id: int = -100,
        require_image: bool = True,
        add_cls_token: bool = True,
        add_sep_token: bool = True,
    ) -> None:
        if require_image:
            assert (
                image_processor is not None
            ), "image_processor must be provided if require_image is True"
        self.require_image = require_image
        self.image_processor = image_processor

        assert (
            padding in self.PADDING_TYPE
        ), f"invalid padding type {padding}, must be in {self.PADDING_TYPE}"
        if padding == "max_length":
            assert max_length > 0, f"invalid max_length {max_length}, must be positive"
        self.padding = padding
        if padding == "longest":
            self.max_length = None
        else:
            self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.add_cls_token = add_cls_token
        self.add_sep_token = add_sep_token

        self.tokenizer = tokenizer

    def __call__(self, features):
        if self.require_image:
            images = []
            for feature in features:
                images.append(Image.open(feature["image_path"]).convert("RGB"))

        line_extraction_spots_list = [
            feature["line_extraction_matrix_spots"] for feature in features
        ]
        ent_linking_head_rel_matrix_spots = [
            feature["ent_linking_head_rel_matrix_spots"] for feature in features
        ]
        ent_linking_tail_rel_matrix_spots = [
            feature["ent_linking_tail_rel_matrix_spots"] for feature in features
        ]
        line_grouping_head_rel_matrix_spots = [
            feature["line_grouping_head_rel_matrix_spots"] for feature in features
        ]
        line_grouping_tail_rel_matrix_spots = [
            feature["line_grouping_tail_rel_matrix_spots"] for feature in features
        ]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        has_labels = "labels" in features[0]
        has_bbox_input = "bbox" in features[0]
        has_orig_bbox_input = "orig_bbox" in features[0]
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            if has_labels:
                batch["labels"] = [
                    label + [self.label_pad_token_id] * (sequence_length - len(label))
                    for label in batch["labels"]
                ]
            if has_bbox_input:
                batch["bbox"] = [
                    bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox))
                    for bbox in batch["bbox"]
                ]
            if has_orig_bbox_input:
                batch["orig_bbox"] = [
                    bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox))
                    for bbox in batch["orig_bbox"]
                ]
        else:
            if has_labels:
                batch["labels"] = [
                    [self.label_pad_token_id] * (sequence_length - len(label)) + label
                    for label in batch["labels"]
                ]
            if has_bbox_input:
                batch["bbox"] = [
                    [[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox
                    for bbox in batch["bbox"]
                ]
            if has_orig_bbox_input:
                batch["orig_bbox"] = [
                    [[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox
                    for bbox in batch["orig_bbox"]
                ]

        valid_sequence_length = sequence_length
        if self.add_cls_token:
            valid_sequence_length -= 1
        shaking_ind2matrix_ind = [
            (ind, end_ind)
            for ind in range(valid_sequence_length)
            for end_ind in list(range(valid_sequence_length))[ind:]
        ]
        matrix_ind2shaking_ind = [
            [0 for _ in range(valid_sequence_length)]
            for _ in range(valid_sequence_length)
        ]
        for shaking_ind, matrix_ind in enumerate(shaking_ind2matrix_ind):
            matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind
        batch["line_extraction_shaking_tag"] = (
            HandshakingTaggingScheme.spots2shaking_tag4batch(
                batch_spots=line_extraction_spots_list,
                shaking_ind2matrix_ind=shaking_ind2matrix_ind,
                matrix_ind2shaking_ind=matrix_ind2shaking_ind,
            )
        )
        batch["ent_linking_head_rel_shaking_tag"] = (
            HandshakingTaggingScheme.spots2shaking_tag4batch(
                batch_spots=ent_linking_head_rel_matrix_spots,
                shaking_ind2matrix_ind=shaking_ind2matrix_ind,
                matrix_ind2shaking_ind=matrix_ind2shaking_ind,
            )
        )
        batch["ent_linking_tail_rel_shaking_tag"] = (
            HandshakingTaggingScheme.spots2shaking_tag4batch(
                batch_spots=ent_linking_tail_rel_matrix_spots,
                shaking_ind2matrix_ind=shaking_ind2matrix_ind,
                matrix_ind2shaking_ind=matrix_ind2shaking_ind,
            )
        )
        batch["line_grouping_head_rel_shaking_tag"] = (
            HandshakingTaggingScheme.spots2shaking_tag4batch(
                batch_spots=line_grouping_head_rel_matrix_spots,
                shaking_ind2matrix_ind=shaking_ind2matrix_ind,
                matrix_ind2shaking_ind=matrix_ind2shaking_ind,
            )
        )
        batch["line_grouping_tail_rel_shaking_tag"] = (
            HandshakingTaggingScheme.spots2shaking_tag4batch(
                batch_spots=line_grouping_tail_rel_matrix_spots,
                shaking_ind2matrix_ind=shaking_ind2matrix_ind,
                matrix_ind2shaking_ind=matrix_ind2shaking_ind,
            )
        )
        for k in [
            "line_extraction_matrix_spots",
            "ent_linking_head_rel_matrix_spots",
            "ent_linking_tail_rel_matrix_spots",
            "line_grouping_head_rel_matrix_spots",
            "line_grouping_tail_rel_matrix_spots",
        ]:
            batch.data.pop(k)

        for k, v in batch.items():
            if isinstance(v[0], list):
                if k not in self.NO_BATCH_KEYS and k not in self.NO_TENSOR_KEYS:
                    batch[k] = torch.tensor(v, dtype=torch.int64)
                elif k not in self.NO_TENSOR_KEYS:
                    batch[k] = [torch.tensor(vv, dtype=torch.int64) for vv in v]
                else:
                    batch[k] = v
            else:
                batch[k] = v

        if self.require_image:
            batch["image"] = self.image_processor(images, return_tensors="pt")[
                "pixel_values"
            ]

        return batch
