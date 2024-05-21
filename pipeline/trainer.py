import collections
import logging
import time
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
from transformers.file_utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import PredictionOutput, speed_metrics
from transformers.utils import logging

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)


class EvalPredictionWithID(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    file_ids: List[str]


class PEneoTrainer(Trainer):
    def __init__(
        self,
        downstream_speedup_ratio: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_names = []

        self.downstream_speedup_ratio = downstream_speedup_ratio

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
        return outputs

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info(
                "Detected the deepspeed argument but it will not be used for evaluation"
            )

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info(f"***** Running {description} *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader

        file_ids = []
        orig_bboxes = []
        text = []
        line_extraction_shaking_outputs = []
        ent_linking_h2h_shaking_outputs = []
        ent_linking_t2t_shaking_outputs = []
        line_grouping_h2h_shaking_outputs = []
        line_grouping_t2t_shaking_outputs = []
        line_extraction_shaking_tags = []
        ent_linking_h2h_shaking_tags = []
        ent_linking_t2t_shaking_tags = []
        line_grouping_h2h_shaking_tags = []
        line_grouping_t2t_shaking_tags = []
        gt_relations = []
        for inputs in dataloader:
            outputs = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

            curr_orig_bbox = outputs.orig_bbox.tolist()
            orig_bboxes += curr_orig_bbox

            if "text" in inputs.keys():
                curr_text = inputs.get("text")
                text += curr_text
            else:
                raise ValueError("No text given in evaluation")

            if "fname" in inputs.keys():
                file_ids += inputs.get("fname")

            line_extraction_shaking_outputs += outputs.line_extraction_shaking_outputs
            ent_linking_h2h_shaking_outputs += outputs.ent_linking_h2h_shaking_outputs
            ent_linking_t2t_shaking_outputs += outputs.ent_linking_t2t_shaking_outputs
            line_grouping_h2h_shaking_outputs += (
                outputs.line_grouping_h2h_shaking_outputs
            )
            line_grouping_t2t_shaking_outputs += (
                outputs.line_grouping_t2t_shaking_outputs
            )
            line_extraction_shaking_tags += inputs.get("line_extraction_shaking_tag")
            ent_linking_h2h_shaking_tags += inputs.get(
                "ent_linking_head_rel_shaking_tag"
            )
            ent_linking_t2t_shaking_tags += inputs.get(
                "ent_linking_tail_rel_shaking_tag"
            )
            line_grouping_h2h_shaking_tags += inputs.get(
                "line_grouping_head_rel_shaking_tag"
            )
            line_grouping_t2t_shaking_tags += inputs.get(
                "line_grouping_tail_rel_shaking_tag"
            )
            gt_relations += inputs.get("relations")

            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )

        _metrics = self.compute_metrics(
            EvalPredictionWithID(
                predictions=(
                    line_extraction_shaking_outputs,
                    ent_linking_h2h_shaking_outputs,
                    ent_linking_t2t_shaking_outputs,
                    line_grouping_h2h_shaking_outputs,
                    line_grouping_t2t_shaking_outputs,
                ),
                label_ids=(
                    line_extraction_shaking_tags,
                    ent_linking_h2h_shaking_tags,
                    ent_linking_t2t_shaking_tags,
                    line_grouping_h2h_shaking_tags,
                    line_grouping_t2t_shaking_tags,
                    gt_relations,
                    orig_bboxes,
                    text,
                ),
                file_ids=file_ids,
            ),
            int(self.state.epoch) if self.state.epoch is not None else 0,
        )

        _metrics[f"{metric_key_prefix}_loss"] = outputs.loss.mean().item()
        _metrics[f"{metric_key_prefix}_line_extraction_loss"] = (
            outputs.line_extraction_loss.mean().item()
        )
        _metrics[f"{metric_key_prefix}_ent_linking_h2h_loss"] = (
            outputs.ent_linking_h2h_loss.mean().item()
        )
        _metrics[f"{metric_key_prefix}_ent_linking_t2t_loss"] = (
            outputs.ent_linking_t2t_loss.mean().item()
        )
        _metrics[f"{metric_key_prefix}_line_grouping_h2h_loss"] = (
            outputs.line_grouping_h2h_loss.mean().item()
        )
        _metrics[f"{metric_key_prefix}_line_grouping_h2h_loss"] = (
            outputs.line_grouping_t2t_loss.mean().item()
        )

        metrics = {}

        # # Prefix all keys with metric_key_prefix + '_'
        for key in list(_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = _metrics.pop(key)
            else:
                metrics[f"{key}"] = _metrics.pop(key)

        return metrics

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        if eval_dataset is not None and not isinstance(
            eval_dataset, collections.abc.Sized
        ):
            raise ValueError("eval_dataset must implement __len__")

        self.args.local_rank = -1
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.args.local_rank = torch.distributed.get_rank()

        start_time = time.time()

        metrics = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            speedup_parameters = [
                name
                for name in get_parameter_names(self.model, [])
                if "peneo_decoder" in name
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in decay_parameters and n in speedup_parameters
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate * self.downstream_speedup_ratio,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in decay_parameters and n in speedup_parameters
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate * self.downstream_speedup_ratio,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n in decay_parameters and n not in speedup_parameters
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if n not in decay_parameters and n not in speedup_parameters
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
