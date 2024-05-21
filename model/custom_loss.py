import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossRandomSample(nn.CrossEntropyLoss):
    def __init__(
        self,
        sample_list: List,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self.sample_list = sample_list
        if sample_list is not None:
            assert (
                len(sample_list) >= 2
            ), f"sample list must contains at least two elements, {len(sample_list)} given"
            self.num_categories = len(sample_list)
        else:
            self.num_categories = None

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.sample_list is None:
            return F.cross_entropy(
                input.float(),
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )

        ce_loss = F.cross_entropy(
            input.float(),
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        mask_list = []
        if self.num_categories == 2 and input.shape[1] >= 2:
            mask_list = [(target == 0), (target != 0)]
        else:
            assert (
                self.num_categories == input.shape[1]
            ), f"shape mismatch, number of elements in sample_list must be 2 or equals dimensions \
                of input, {self.num_categories} and {input.shape[1]} given"
            for cate_index in range(self.num_categories):
                mask_list.append(target == cate_index)

        loss_list = []
        num_keep_list = []
        for index, mask in enumerate(mask_list):
            curr_sample = self.sample_list[index]
            curr_loss = ce_loss[mask]
            num_keep = min(curr_sample, curr_loss.shape[0])
            num_keep_list.append(num_keep)

            if num_keep == curr_sample:
                keep_index = random.sample(range(int(curr_loss.shape[0])), num_keep)
                keep_index = torch.tensor(keep_index, device=curr_loss.device)
                keep_loss = curr_loss[keep_index]
            else:
                keep_loss = curr_loss

            loss_list.append(keep_loss)

        if self.reduction == "sum":
            keep_ce_loss = torch.zeros((1,), dtype=float, device=target.device)
            for loss in loss_list:
                keep_ce_loss = keep_ce_loss + loss.sum()
        elif self.reduction == "mean":
            num_keep_total = 0
            keep_ce_loss = torch.zeros((1,), dtype=float, device=target.device)
            for loss, num_keep in zip(loss_list, num_keep_list):
                keep_ce_loss = keep_ce_loss + loss.sum()
                num_keep_total = num_keep_total + num_keep
            keep_ce_loss /= num_keep_total
        elif self.reduction == "none":
            keep_ce_loss = torch.stack(loss_list)
        else:
            raise ValueError(
                f"the given reduction value {self.reduction} is invalid, must be 'none', 'mean' or 'sum' "
            )

        return keep_ce_loss


class CrossEntropyLossOHEM(nn.CrossEntropyLoss):
    """apply online hard example mining to cross entropy loss

    Parameters
    ----------
    hard_positive_ratio : float
        ratio of hard positive samples to keep, if not given, num_hard_positive must be given
    hard_negative_ratio : float
        ratio of hard negative samples to keep, if not given, num_hard_negative must be given
    num_hard_positive : int
        number of hard positive samples to keep, if not given, hard_positive_ratio must be given.
        default to -1, which means no OHEM.
    num_hard_negative : int
        number of hard negative samples to keep, if not given, hard_negative_ratio must be given.
        default to -1, which means no OHEM.
    weight : torch.Tensor
        weight of each class, default to None
    size_average : bool
        whether to average the loss, default to None
    ignore_index : int
        label index to ignore when calculating loss, default to -100
    reduction : str
        reduction method, must be 'none', 'mean' or 'sum', default to 'mean'
    label_smoothing : float
        label smoothing factor, default to 0.0
    random : bool
        whether to randomly sample examples before applying OHEM, default to False
        if True, 2 * num_hard_positive or num_hard_negative samples will be randomly selected,
        then OHEM applied to these selected samples
    """

    def __init__(
        self,
        hard_positive_ratio: float = None,
        hard_negative_ratio: float = None,
        num_hard_positive: int = -1,
        num_hard_negative: int = -1,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduction: str = "mean",
        random: bool = False,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduction=reduction,
        )

        if not hard_positive_ratio:
            hard_positive_ratio = None
        if not hard_negative_ratio:
            hard_negative_ratio = None

        if hard_positive_ratio is not None:
            assert (int(hard_positive_ratio * 1000) >= 0) and (
                int(hard_positive_ratio * 1000) <= 1000
            ), f"hard_positive_ratio must be in [0, 1], {hard_positive_ratio} given"
            self.num_hard_positive = None
            self.hard_positive_ratio = hard_negative_ratio
        elif num_hard_positive is not None:
            self.num_hard_positive = num_hard_positive
            self.hard_positive_ratio = None
        else:
            raise ValueError(
                "either num_hard_positive or hard_positive_ratio must be given"
            )

        if hard_negative_ratio is not None:
            assert (int(hard_negative_ratio * 1000) >= 0) and (
                int(hard_negative_ratio * 1000) <= 1000
            ), f"hard_negative_ratio must be in [0, 1], {hard_negative_ratio} given"
            self.num_hard_negative = None
            self.hard_negative_ratio = hard_negative_ratio
        elif num_hard_negative is not None:
            self.num_hard_negative = num_hard_negative
            self.hard_negative_ratio = None
        else:
            raise ValueError(
                "either num_hard_negative or hard_negative_ratio must be given"
            )

        self.random = random

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if (
            self.num_hard_positive == -1
            and self.num_hard_negative == -1
            and self.hard_positive_ratio is None
            and self.hard_negative_ratio is None
        ):
            return F.cross_entropy(
                input.float(),
                target,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )

        ce_loss = F.cross_entropy(
            input.float(),
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )

        if self.hard_positive_ratio is not None:
            num_ele = target.view(-1).shape[0]
            hp = int(num_ele * self.hard_positive_ratio)
            if hp != 0:
                self.num_hard_positive = hp
            else:
                self.num_hard_positive = num_ele
                print(
                    "hard_positive_ratio is too small, set to num_ele. Check your config"
                )

        if self.hard_negative_ratio is not None:
            num_ele = target.view(-1).shape[0]
            hn = int(num_ele * self.hard_negative_ratio)
            if hn != 0:
                self.num_hard_negative = hn
            else:
                self.num_hard_negative = num_ele
                print(
                    "hard_negative_ratio is too small, set to num_ele. Check your config"
                )

        mask = target == 0
        positive_loss = ce_loss[~mask]
        negative_loss = ce_loss[mask]

        if self.random:
            num_random_pos = 2 * self.num_hard_positive
            if num_random_pos < positive_loss.shape[0]:
                keep_index = random.sample(
                    range(int(positive_loss.shape[0])), num_random_pos
                )
                keep_index = torch.tensor(keep_index, device=ce_loss.device)
                positive_loss = positive_loss[keep_index]

            num_random_neg = 2 * self.num_hard_negative
            if num_random_neg < negative_loss.shape[0]:
                keep_index = random.sample(
                    range(int(negative_loss.shape[0])), num_random_neg
                )
                keep_index = torch.tensor(keep_index, device=ce_loss.device)
                negative_loss = negative_loss[keep_index]

        sorted_positive_loss, sorted_positive_index = torch.sort(
            positive_loss, descending=True
        )
        num_positive_keep = min(sorted_positive_loss.shape[0], self.num_hard_positive)
        if num_positive_keep <= 0:
            pass
        elif num_positive_keep < sorted_positive_loss.shape[0]:
            keep_pos_index = sorted_positive_index[:num_positive_keep]
            sorted_positive_loss = sorted_positive_loss[keep_pos_index]

        sorted_negative_loss, sorted_negative_index = torch.sort(
            negative_loss, descending=True
        )
        num_negative_keep = min(sorted_negative_loss.shape[0], self.num_hard_negative)
        if num_negative_keep <= 0:
            pass
        elif num_negative_keep < sorted_negative_loss.shape[0]:
            keep_neg_index = sorted_negative_index[:num_negative_keep]
            sorted_negative_loss = sorted_negative_loss[keep_neg_index]

        if self.reduction == "sum":
            keep_ce_loss = sorted_positive_loss.sum() + sorted_negative_loss.sum()
        elif self.reduction == "mean":
            keep_ce_loss = (sorted_positive_loss.sum() + sorted_negative_loss.sum()) / (
                num_positive_keep + num_negative_keep
            )
        elif self.reduction == "none":
            keep_ce_loss = torch.stack([sorted_positive_loss, sorted_negative_loss])
        else:
            raise ValueError(
                f"the given reduction value {self.reduction} is invalid, must be 'none', 'mean' or 'sum' "
            )

        return keep_ce_loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss
