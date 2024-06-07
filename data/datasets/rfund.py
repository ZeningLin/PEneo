import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast, ProcessorMixin

from ..data_utils import box_augmentation, normalize_bbox, sort_boxes, string_f2h


@dataclass
class LineInfo:
    center_x: int
    center_y: int
    coords: List[float]
    tokens: List[str]
    sos_processed_tokens: List[str]
    category: str
    entity_first_line: bool
    orig_entity_id: str
    orig_line_id: str
    orig_next_line: int = None
    sorted_start_token: int = None
    sorted_end_token: int = None


class RFUNDDataset(Dataset):
    """Dataset class for RFUND"""

    LANG_LIST = ["en", "zh", "ja", "es", "fr", "de", "it", "pt"]
    SPLIT_LIST = ["train", "dev", "test"]

    ENTITY_LABEL_LIST = ["other", "header", "question", "answer"]
    LABEL_LIST = [
        "O",
        "B-header",
        "I-header",
        "B-question",
        "I-question",
        "B-answer",
        "I-answer",
    ]
    LABEL_NAME2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
    LABEL_ID2NAME = {idx: label for idx, label in enumerate(LABEL_LIST)}

    def __init__(
        self,
        data_root: str,
        split: str,
        language: str,
        tokenizer: Union[PreTrainedTokenizerFast, ProcessorMixin, str],
        tokenizer_fetcher: Optional[Callable] = None,
        max_token_len: int = 511,
        add_cls_token: bool = False,
        add_sep_token: bool = False,
        apply_box_aug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        assert (
            language in self.LANG_LIST
        ), f"Language {language} not supported, should be one of {self.LANG_LIST}"
        self.language = language

        assert (
            split in self.SPLIT_LIST
        ), f"Split {split} not supported, should be one of {self.SPLIT_LIST}"
        self.split = split

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        elif isinstance(tokenizer, ProcessorMixin):
            self.tokenizer = tokenizer.tokenizer
        else:
            self.tokenizer = tokenizer

        self.tokenizer_fetcher = tokenizer_fetcher

        self.image_root = os.path.join(data_root, "images", language)
        if split in ["dev", "test"]:
            split = "val"
        self.annotation_dir = os.path.join(data_root, f"{language}.{split}.json")

        annotation = json.load(open(self.annotation_dir, "r", encoding="utf-8"))
        self.annotation = annotation["documents"]

        self.max_token_len = max_token_len
        self.add_cls_token = add_cls_token
        self.add_sep_token = add_sep_token

        self.apply_box_aug = apply_box_aug

    def __len__(self) -> int:
        return len(self.annotation)

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

        return string_f2h(line_text)

    def __getitem__(self, index):
        document_info = self.annotation[index]
        image_fname = document_info["img"]["fname"]
        image_dir = os.path.join(self.image_root, image_fname)
        image_w, image_h = (
            document_info["img"]["width"],
            document_info["img"]["height"],
        )

        all_orig_line_list: List[LineInfo] = []
        all_orig_box_list = []
        empty_line = set()
        empty_entity = set()
        entity_id_to_text_map = {}
        entity_first_line_map = {}
        entity_last_line_map = {}
        line_id_to_entity_id_map = {}
        for entity_info in document_info["entities"]:
            first_line_flag = True
            entity_text_list = []
            for line_info in entity_info["lines"]:
                line_text: str = line_info["text"]
                if not first_line_flag and self.language not in ["zh", "ja"]:
                    line_text = " " + line_text
                line_text = self._special_text_replace(line_text)

                line_tokens = self.tokenizer.tokenize(line_text)
                if self.tokenizer_fetcher is not None:
                    line_sos_processed_tokens = self.tokenizer_fetcher(
                        line_text, line_tokens
                    )
                else:
                    line_sos_processed_tokens = line_tokens

                if len(line_sos_processed_tokens) == 0:
                    empty_line.add(line_info["id"])
                    continue

                entity_text_list.append(line_text)
                line_left, line_top, line_right, line_bottom = line_info["bbox"]
                if self.apply_box_aug:
                    (
                        line_left,
                        line_top,
                        line_right,
                        line_bottom,
                    ) = box_augmentation(
                        (line_left, line_right, line_top, line_bottom),
                        image_w,
                        image_h,
                    )
                    if line_left >= line_right:
                        if line_right == 0:
                            line_left, line_right = 0, 1
                        else:
                            line_left = line_right - 1
                    if line_top >= line_bottom:
                        if line_bottom == 0:
                            line_top, line_bottom = 0, 1
                        else:
                            line_top = line_bottom - 1

                line_center_x = (line_left + line_right) / 2
                line_center_y = (line_top + line_bottom) / 2

                all_orig_line_list.append(
                    LineInfo(
                        center_x=line_center_x,
                        center_y=line_center_y,
                        coords=[
                            line_left,
                            line_top,
                            line_right,
                            line_bottom,
                        ],
                        tokens=line_tokens,
                        sos_processed_tokens=line_sos_processed_tokens,
                        category=entity_info["label"],
                        entity_first_line=first_line_flag,
                        orig_entity_id=entity_info["id"],
                        orig_line_id=line_info["id"],
                    )
                )
                all_orig_box_list.append([line_left, line_top, line_right, line_bottom])
                if first_line_flag:
                    entity_first_line_map[entity_info["id"]] = line_info["id"]
                first_line_flag = False

                line_id_to_entity_id_map[line_info["id"]] = entity_info["id"]

            if len(all_orig_line_list) == 0:
                empty_entity.add(entity_info["id"])
                continue

            if len(all_orig_line_list) > 0:
                entity_last_line_map[entity_info["id"]] = all_orig_line_list[
                    -1
                ].orig_line_id
            entity_id_to_text_map[entity_info["id"]] = "".join(entity_text_list)

        ro_sorted_box_idx = sort_boxes(all_orig_box_list)
        all_sorted_line_list: List[LineInfo] = [
            all_orig_line_list[i] for i in ro_sorted_box_idx
        ]

        texts = []
        line_extraction_matrix_spots = []
        ent_linking_head_rel_matrix_spots = []
        ent_linking_tail_rel_matrix_spots = []
        line_grouping_head_rel_matrix_spots = []
        line_grouping_tail_rel_matrix_spots = []

        input_ids = []
        bbox = []
        orig_bbox = []

        curr_token_idx = 0
        line_orig_to_sorted_map: Dict[int, int] = {}
        in_scope_entity_id = set()
        in_scope_line_id = set()
        for sorted_line_idx, ln in enumerate(all_sorted_line_list):
            line_orig_to_sorted_map.update({ln.orig_line_id: sorted_line_idx})
            (
                line_class_str,
                curr_line_orig_bbox,
                line_tokens,
                line_sos_processed_tokens,
            ) = (
                ln.category,
                ln.coords,
                ln.tokens,
                ln.sos_processed_tokens,
            )
            line_tokens: List[str]
            line_token_ids: List[int] = self.tokenizer.convert_tokens_to_ids(
                line_tokens
            )

            line_token_len = len(line_token_ids)
            if curr_token_idx + line_token_len >= self.max_token_len:
                # reach max token length, break
                break

            in_scope_entity_id.add(ln.orig_entity_id)
            # part of the entity lines may fall out of scope
            # but the corresponding entity is still recorded as in scope
            in_scope_line_id.add(ln.orig_line_id)

            curr_line_norm_bbox = normalize_bbox(
                curr_line_orig_bbox, (image_w, image_h)
            )

            orig_bbox.extend([curr_line_orig_bbox] * line_token_len)
            bbox.extend([curr_line_norm_bbox] * line_token_len)
            texts.extend(line_sos_processed_tokens)
            input_ids.extend(line_token_ids)

            sorted_line_start_token = curr_token_idx
            all_sorted_line_list[sorted_line_idx].sorted_start_token = (
                sorted_line_start_token
            )
            curr_token_idx += line_token_len
            sorted_line_end_token = curr_token_idx
            all_sorted_line_list[sorted_line_idx].sorted_end_token = (
                sorted_line_end_token
            )

            if line_class_str == "question" or line_class_str == "answer":
                line_extraction_matrix_spots.append(
                    (sorted_line_start_token, sorted_line_end_token - 1, 1)
                )

        for kv_entity_info in document_info["relations"]["kv_entity"]:
            question_entity_id = kv_entity_info["from_id"]
            answer_entity_id = kv_entity_info["to_id"]
            if question_entity_id in empty_entity or answer_entity_id in empty_entity:
                continue
            if (question_entity_id not in in_scope_entity_id) or (
                answer_entity_id not in in_scope_entity_id
            ):
                continue

            question_first_line_orig_id = entity_first_line_map[question_entity_id]
            answer_first_line_orig_id = entity_first_line_map[answer_entity_id]
            question_last_line_orig_id = entity_last_line_map[question_entity_id]
            answer_last_line_orig_id = entity_last_line_map[answer_entity_id]

            if (
                (question_first_line_orig_id not in in_scope_line_id)
                or (question_last_line_orig_id not in in_scope_line_id)
                or (answer_first_line_orig_id not in in_scope_line_id)
                or (answer_last_line_orig_id not in in_scope_line_id)
            ):
                continue

            question_first_line_sorted_id = line_orig_to_sorted_map[
                question_first_line_orig_id
            ]
            answer_first_line_sorted_id = line_orig_to_sorted_map[
                answer_first_line_orig_id
            ]
            question_last_line_sorted_id = line_orig_to_sorted_map[
                question_last_line_orig_id
            ]
            answer_last_line_sorted_id = line_orig_to_sorted_map[
                answer_last_line_orig_id
            ]

            question_first_line_start_token = all_sorted_line_list[
                question_first_line_sorted_id
            ].sorted_start_token
            answer_first_line_start_token = all_sorted_line_list[
                answer_first_line_sorted_id
            ].sorted_start_token
            question_last_line_end_token = all_sorted_line_list[
                question_last_line_sorted_id
            ].sorted_end_token
            answer_last_line_end_token = all_sorted_line_list[
                answer_last_line_sorted_id
            ].sorted_end_token

            if question_first_line_start_token < answer_first_line_start_token:
                ent_linking_head_rel_matrix_spots.append(
                    (
                        question_first_line_start_token,
                        answer_first_line_start_token,
                        1,
                    )
                )
            else:
                ent_linking_head_rel_matrix_spots.append(
                    (
                        answer_first_line_start_token,
                        question_first_line_start_token,
                        2,
                    )
                )

            if question_last_line_end_token < answer_last_line_end_token:
                ent_linking_tail_rel_matrix_spots.append(
                    (
                        question_last_line_end_token - 1,
                        answer_last_line_end_token - 1,
                        1,
                    )
                )
            else:
                ent_linking_tail_rel_matrix_spots.append(
                    (
                        answer_last_line_end_token - 1,
                        question_last_line_end_token - 1,
                        2,
                    )
                )

        for line_linking_info in document_info["relations"]["line_grouping"]:
            from_line_orig_id = line_linking_info["from_id"]
            to_line_orig_id = line_linking_info["to_id"]

            if (from_line_orig_id in empty_line) or (to_line_orig_id in empty_line):
                continue

            from_line_entity_id, to_line_entity_id = (
                line_id_to_entity_id_map.get(from_line_orig_id, -1),
                line_id_to_entity_id_map.get(to_line_orig_id, -1),
            )
            if (from_line_entity_id not in in_scope_entity_id) or (
                to_line_entity_id not in in_scope_entity_id
            ):
                continue

            if (from_line_orig_id not in in_scope_line_id) or (
                to_line_orig_id not in in_scope_line_id
            ):
                continue

            from_line_sorted_id = line_orig_to_sorted_map[from_line_orig_id]
            to_line_sorted_id = line_orig_to_sorted_map[to_line_orig_id]

            from_line_start_token = all_sorted_line_list[
                from_line_sorted_id
            ].sorted_start_token
            from_line_end_token = all_sorted_line_list[
                from_line_sorted_id
            ].sorted_end_token
            to_line_start_token = all_sorted_line_list[
                to_line_sorted_id
            ].sorted_start_token
            to_line_end_token = all_sorted_line_list[to_line_sorted_id].sorted_end_token

            if (
                from_line_start_token is None
                or from_line_end_token is None
                or to_line_start_token is None
                or to_line_end_token is None
            ):
                continue

            if from_line_start_token < to_line_start_token:
                line_grouping_head_rel_matrix_spots.append(
                    (from_line_start_token, to_line_start_token, 1)
                )
            else:
                line_grouping_head_rel_matrix_spots.append(
                    (to_line_start_token, from_line_start_token, 2)
                )

            if from_line_end_token < to_line_end_token:
                line_grouping_tail_rel_matrix_spots.append(
                    (from_line_end_token - 1, to_line_end_token - 1, 1)
                )
            else:
                line_grouping_tail_rel_matrix_spots.append(
                    (to_line_end_token - 1, from_line_end_token - 1, 2)
                )

        relations = []
        for kv_entity_info in document_info["relations"]["kv_entity"]:
            question_id = kv_entity_info["from_id"]
            answer_id = kv_entity_info["to_id"]
            if (
                question_id not in entity_id_to_text_map.keys()
                or answer_id not in entity_id_to_text_map.keys()
                or question_id in empty_entity
                or answer_id in empty_entity
                or question_id not in in_scope_entity_id
                or answer_id not in in_scope_entity_id
            ):
                continue
            question_text = entity_id_to_text_map[question_id]
            answer_text = entity_id_to_text_map[answer_id]
            relations.append({"key": question_text, "value": answer_text})

        if self.add_cls_token:
            input_ids = [self.tokenizer.cls_token_id] + input_ids
            bbox = [[0, 0, 0, 0]] + bbox
            orig_bbox = [[0, 0, 0, 0]] + orig_bbox
        if self.add_sep_token:
            input_ids = input_ids + [self.tokenizer.sep_token_id]
            bbox = bbox + [[0, 0, 0, 0]]
            orig_bbox = orig_bbox + [[0, 0, 0, 0]]

        assert len(input_ids) == len(bbox), f"bbox_length mismatch {image_fname}"
        assert len(input_ids) == len(
            orig_bbox
        ), f"orig_bbox length mismatch {image_fname}"
        assert len(ent_linking_head_rel_matrix_spots) == len(
            ent_linking_tail_rel_matrix_spots
        ), f"entity relation length mismatch {image_fname}"
        assert len(line_grouping_head_rel_matrix_spots) == len(
            line_grouping_tail_rel_matrix_spots
        ), f"line relation length mismatch {image_fname}"

        return {
            "fname": image_fname,
            "image_path": image_dir,
            "input_ids": input_ids,
            "bbox": bbox,
            "orig_bbox": orig_bbox,
            "text": texts,
            "relations": relations,
            "line_extraction_matrix_spots": line_extraction_matrix_spots,
            "ent_linking_head_rel_matrix_spots": ent_linking_head_rel_matrix_spots,
            "ent_linking_tail_rel_matrix_spots": ent_linking_tail_rel_matrix_spots,
            "line_grouping_head_rel_matrix_spots": line_grouping_head_rel_matrix_spots,
            "line_grouping_tail_rel_matrix_spots": line_grouping_tail_rel_matrix_spots,
        }
