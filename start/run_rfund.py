import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import transformers
from transformers import HfArgumentParser, ProcessorMixin, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from data import DataCollatorForPEneo, RFUNDDataset
from model import PEneoConfig, PEneoModel
from model.backbone_mapping import BACKBONE_MAPPING
from model.peneo_decoder import HandshakingTaggingScheme
from pipeline.decode import decode_peneo
from pipeline.evaluation import calculate_detail_KVPE_metric, calculate_KVPE_metric
from pipeline.trainer import PEneoTrainer

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
    data_dir: str = field(
        metadata={"help": "Path to data directory"},
    )
    dataset_name: str = field(
        default="auto",
        metadata={
            "help": "Name of the dataset, if set to auto, will infer from data_dir"
        },
    )
    language: str = field(
        default="en",
        metadata={
            "help": "Language of the data, used in the case of multi-lingual dataset"
        },
    )
    apply_box_aug: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply box augmentation to the document OCR boxes"
        },
    )
    detail_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to perform detailed evaluation including line-level SER and linking metrics"
        },
    )
    save_eval_detail: bool = field(
        default=False,
        metadata={"help": "Whether to save detailed evaluation result"},
    )
    start_eval_epoch: int = field(
        default=0,
        metadata={"help": "Start evaluation from this epoch"},
    )


def convert_hf_args_to_dict(args: Any) -> Any:
    """
    Convert HfArgumentParser args to dict in an iterative manner,
    use for saving args to json file.

    Parameters
    ----------
    args : Any
        HfArgumentParser args

    Returns
    -------
    Convert args in dict format.
    """
    if args is None:
        return None
    if not hasattr(args, "__dict__"):
        return None

    _training_args_dict = {
        k: v for k, v in vars(args).items() if not k.startswith("__")
    }
    training_args_dict = {}
    for k, v in _training_args_dict.items():
        if isinstance(v, (int, float, str, bool, list, dict, tuple)):
            training_args_dict[k] = v
        else:
            training_args_dict[k] = convert_hf_args_to_dict(v)

    return training_args_dict


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=(
            logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
        ),
    )
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 1:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Save args
    all_args = {
        "model_args": convert_hf_args_to_dict(model_args),
        "data_args": convert_hf_args_to_dict(data_args),
        "training_args": convert_hf_args_to_dict(training_args),
    }
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, "args.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(all_args, f, indent=4)

    # Load Model
    config = PEneoConfig.from_pretrained(
        (
            model_args.config_name
            if model_args.config_name
            else model_args.model_name_or_path
        ),
    )
    backbone_info = BACKBONE_MAPPING.get(config.backbone_name, None)
    if backbone_info is None:
        logger.error(
            f"Invalid backbone name {config.backbone_name}",
            f"Available backbones are {list(BACKBONE_MAPPING.keys())}",
        )
        raise ValueError()

    processor = backbone_info.processor.from_pretrained(
        model_args.model_name_or_path, use_fast=True
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

    # Load dataset
    if training_args.do_train:
        train_dataset = RFUNDDataset(
            data_root=data_args.data_dir,
            split="train",
            language=data_args.language,
            tokenizer=tokenizer,
            tokenizer_fetcher=backbone_info.tokenizer_fetcher,
            max_token_len=backbone_info.max_token_len,
            add_cls_token=backbone_info.add_cls_token,
            add_sep_token=backbone_info.add_sep_token,
            apply_box_aug=data_args.apply_box_aug,
        )
    if training_args.do_eval:
        eval_dataset = RFUNDDataset(
            data_root=data_args.data_dir,
            split="dev",
            language=data_args.language,
            tokenizer=tokenizer,
            tokenizer_fetcher=backbone_info.tokenizer_fetcher,
            max_token_len=backbone_info.max_token_len,
            add_cls_token=backbone_info.add_cls_token,
            add_sep_token=backbone_info.add_sep_token,
            apply_box_aug=data_args.apply_box_aug,
        )

    collator = DataCollatorForPEneo(
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_length=backbone_info.max_token_len,
        require_image=backbone_info.image_processor is not None,
        add_cls_token=backbone_info.add_cls_token,
        add_sep_token=backbone_info.add_sep_token,
    )
    handshaking_tagger = HandshakingTaggingScheme()

    # Keep it for multi-backbone compatibility
    training_args.remove_unused_columns = False

    def compute_metrics(p, epoch):
        predictions, label_ids, file_ids = p
        (
            line_extraction_shaking_outputs,
            ent_linking_h2h_shaking_outputs,
            ent_linking_t2t_shaking_outputs,
            line_grouping_h2h_shaking_outputs,
            line_grouping_t2t_shaking_outputs,
        ) = predictions
        (
            line_extraction_shaking_tags,
            ent_linking_h2h_shaking_tags,
            ent_linking_t2t_shaking_tags,
            line_grouping_h2h_shaking_tags,
            line_grouping_t2t_shaking_tags,
            gt_relations,
            orig_bboxes,
            texts,
        ) = label_ids

        if epoch >= data_args.start_eval_epoch:
            all_pred_results, all_gt_results, all_fname = decode_peneo(
                handshaking_tagger=handshaking_tagger,
                texts=texts,
                line_extraction_shaking_outputs=line_extraction_shaking_outputs,
                ent_linking_h2h_shaking_outputs=ent_linking_h2h_shaking_outputs,
                ent_linking_t2t_shaking_outputs=ent_linking_t2t_shaking_outputs,
                line_grouping_h2h_shaking_outputs=line_grouping_h2h_shaking_outputs,
                line_grouping_t2t_shaking_outputs=line_grouping_t2t_shaking_outputs,
                line_extraction_shaking_tags=line_extraction_shaking_tags,
                ent_linking_h2h_shaking_tags=ent_linking_h2h_shaking_tags,
                ent_linking_t2t_shaking_tags=ent_linking_t2t_shaking_tags,
                line_grouping_h2h_shaking_tags=line_grouping_h2h_shaking_tags,
                line_grouping_t2t_shaking_tags=line_grouping_t2t_shaking_tags,
                orig_bboxes=orig_bboxes,
                file_ids=file_ids,
            )

            if data_args.detail_eval:
                metric, detail = calculate_detail_KVPE_metric(
                    all_pred=all_pred_results,
                    all_gt=all_gt_results,
                    all_fname=all_fname,
                )
            else:
                metric, detail = calculate_KVPE_metric(
                    all_pred=all_pred_results,
                    all_gt=all_gt_results,
                    all_fname=all_fname,
                )

            if data_args.save_eval_detail and is_main_process(training_args.local_rank):
                with open(
                    os.path.join(training_args.output_dir, "detail.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(detail, f, ensure_ascii=False, indent=4)
        else:
            metric = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        return metric

    # Initialize Trainer
    trainer = PEneoTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        downstream_speedup_ratio=config.peneo_downstream_speedup_ratio,
    )

    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=last_checkpoint,
        )
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        processor.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
