from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

from .configuration_peneo import PEneoConfig
from .custom_loss import CrossEntropyLossOHEM


class HandshakingTaggingScheme:
    """
    Modified based on the `TPlinker-joint-extraction` repository
    at https://github.com/131250208/TPlinker-joint-extraction
    """

    @staticmethod
    def spots2shaking_tag(spots: List[Tuple], seq_len: int) -> torch.Tensor:
        """
        convert spots to shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entity
        return:
            shake_seq_tag: (shaking_seq_len, )
        """
        matrix_ind2shaking_ind = [[0 for i in range(seq_len)] for j in range(seq_len)]
        shaking_seq_len = seq_len * (seq_len + 1) // 2
        shaking_seq_tag = torch.zeros(shaking_seq_len).long()
        for sp in spots:
            shaking_ind = matrix_ind2shaking_ind[sp[0]][sp[1]]
            shaking_seq_tag[shaking_ind] = sp[2]
        return shaking_seq_tag

    @staticmethod
    def spots2shaking_tag4batch(
        batch_spots: List[Tuple],
        shaking_ind2matrix_ind: List[Tuple] = None,
        matrix_ind2shaking_ind: List[List[int]] = None,
        seq_len: int = None,
    ) -> torch.Tensor:
        """
        convert spots to batch shaking seq tag
        spots: [(start_ind, end_ind, tag_id), ], for entity
        return:
            batch_shake_seq_tag: (batch_size, shaking_seq_len)
        """
        if shaking_ind2matrix_ind is not None and matrix_ind2shaking_ind is not None:
            seq_len = len(matrix_ind2shaking_ind)
        elif seq_len is not None:
            shaking_ind2matrix_ind = [
                (ind, end_ind)
                for ind in range(seq_len)
                for end_ind in list(range(seq_len))[ind:]
            ]
            matrix_ind2shaking_ind = [
                [0 for i in range(seq_len)] for j in range(seq_len)
            ]
            for shaking_ind, matrix_ind in enumerate(shaking_ind2matrix_ind):
                matrix_ind2shaking_ind[matrix_ind[0]][matrix_ind[1]] = shaking_ind
        else:
            raise ValueError(
                f"If shaking_ind2matrix_ind and matrix_ind2shaking_ind are not provided,"
                "seq_len must be given"
            )

        shaking_seq_len = seq_len * (seq_len + 1) // 2
        batch_shaking_seq_tag = torch.zeros(len(batch_spots), shaking_seq_len).long()
        for batch_id, spots in enumerate(batch_spots):
            for sp in spots:
                shaking_ind = matrix_ind2shaking_ind[sp[0]][sp[1]]
                tag_id = sp[2]
                batch_shaking_seq_tag[batch_id][shaking_ind] = tag_id
        return batch_shaking_seq_tag

    @staticmethod
    def get_spots_from_shaking_tag(
        shaking_tag: torch.Tensor, shaking_ind2matrix_ind=None, seq_len: int = None
    ) -> List[Tuple]:
        """
        shaking_tag -> spots
        shaking_tag: (shaking_seq_len, )
        spots: [(start_ind, end_ind, tag_id), ]
        """
        if shaking_ind2matrix_ind is not None:
            pass
        elif seq_len is not None:
            shaking_ind2matrix_ind = [
                (ind, end_ind)
                for ind in range(seq_len)
                for end_ind in list(range(seq_len))[ind:]
            ]
        else:
            raise ValueError(
                f"If shaking_ind2matrix_ind and matrix_ind2shaking_ind are not provided,"
                "seq_len must be given"
            )

        if len(shaking_tag.shape) > 1 and shaking_tag.shape[-1] > 1:
            shaking_tag = shaking_tag.softmax(-1)
            shaking_tag_pred = shaking_tag.argmax(-1)
            shaking_tag_score = torch.max(shaking_tag, dim=-1)[0]
        else:
            shaking_tag_pred = shaking_tag
            shaking_tag_score = torch.ones_like(shaking_tag)
        del shaking_tag

        spots = []
        for shaking_ind in torch.nonzero(shaking_tag_pred):
            shaking_ind_ = shaking_ind[0].item()
            tag_id = shaking_tag_pred[shaking_ind_].item()
            pred_score = shaking_tag_score[shaking_ind_].item()
            matrix_inds = shaking_ind2matrix_ind[shaking_ind_]
            spot = (matrix_inds[0], matrix_inds[1], tag_id, pred_score)
            spots.append(spot)
        return spots


class HandshakingKernel(nn.Module):
    """
    Apply pairwise token concatenation and return the flattened
    upper triangular part of the shaking matrix.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, seq_hiddens: torch.Tensor) -> torch.Tensor:
        """Apply pairwise token concatenation and return the flattened
        upper triangular part of the shaking matrix.

        Parameters
        ----------
        seq_hiddens : torch.Tensor
            (batch_size, seq_len, hidden_size)

        Returns
        -------
        shaking_matrix: torch.Tensor
            (batch_size, (1 + seq_len) * seq_len / 2, hidden_size)
        """
        _, seq_len, _ = seq_hiddens.shape
        shaking_matrix = torch.cat(
            [
                seq_hiddens.unsqueeze(2).repeat(1, 1, seq_len, 1),
                seq_hiddens.unsqueeze(1).repeat(1, seq_len, 1, 1),
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)
        triu_index = torch.triu_indices(seq_len, seq_len, device=seq_hiddens.device)
        triu_index = seq_len * triu_index[0] + triu_index[1]
        shaking_matrix = shaking_matrix.flatten(-2)[..., triu_index]
        shaking_matrix = shaking_matrix.permute(0, 2, 1)
        shaking_matrix = self.activation(self.combine_fc(shaking_matrix))

        return shaking_matrix


@dataclass
class PEneoOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None

    line_extraction_loss: Optional[torch.Tensor] = None
    ent_linking_h2h_loss: Optional[torch.Tensor] = None
    ent_linking_t2t_loss: Optional[torch.Tensor] = None
    line_grouping_h2h_loss: Optional[torch.Tensor] = None
    line_grouping_t2t_loss: Optional[torch.Tensor] = None

    line_extraction_shaking_outputs: Optional[torch.Tensor] = None
    ent_linking_h2h_shaking_outputs: Optional[torch.Tensor] = None
    ent_linking_t2t_shaking_outputs: Optional[torch.Tensor] = None
    line_grouping_h2h_shaking_outputs: Optional[torch.Tensor] = None
    line_grouping_t2t_shaking_outputs: Optional[torch.Tensor] = None

    attentions: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    orig_bbox: Optional[torch.Tensor] = None


class PEneoDecoder(nn.Module):
    """PEneo pair extraction downstream head"""

    def __init__(
        self,
        config: PEneoConfig,
        input_size: int,
    ) -> None:
        super().__init__()
        self.decoder_shrink = config.peneo_decoder_shrink
        backbone_hidden_size = config.backbone_config["hidden_size"]
        backbone_hidden_dropout_prob = config.backbone_config["hidden_dropout_prob"]
        if self.decoder_shrink:
            decoder_hidden_size = backbone_hidden_size // 2
            self.shrink_projection = nn.Sequential(
                nn.Linear(input_size, backbone_hidden_size),
                nn.ReLU(),
                nn.Dropout(backbone_hidden_dropout_prob),
                nn.Linear(backbone_hidden_size, decoder_hidden_size),
                nn.ReLU(),
                nn.Dropout(backbone_hidden_dropout_prob),
            )
        else:
            decoder_hidden_size = input_size

        # handshaking kernel
        self.handshaking_kernel = HandshakingKernel(decoder_hidden_size)

        def build_classifier(
            input_size: int,
            output_size: int,
            mid_size: int = None,
            activation: nn.Module = nn.ReLU(),
        ) -> nn.Module:
            """Construct linear classifier

            Parameters
            ----------
            input_size : int
                input size of the classifier
            output_size : int
                number of output classes
            mid_size : int, optional
                If the number of layers is greater than 1, the hidden size of the classifier
                is set to mid_size. If None, mid_size is set to input_size // 2
                By default None
            activation : nn.Module, optional
                activation function, by default nn.ReLU()

            """
            if config.peneo_classifier_num_layers == 1:
                return nn.Linear(input_size, output_size)

            if mid_size is None:
                mid_size = input_size
            modules = [
                nn.Linear(input_size, mid_size),
                activation,
                nn.Dropout(backbone_hidden_dropout_prob),
            ]
            for _ in range(1, config.peneo_classifier_num_layers - 1):
                modules += [
                    nn.Linear(mid_size, mid_size),
                    activation,
                    nn.Dropout(backbone_hidden_dropout_prob),
                ]
            modules.append(nn.Linear(mid_size, output_size))

            return nn.Sequential(*modules)

        self.line_extraction_fc = build_classifier(
            input_size=decoder_hidden_size,
            output_size=2,
        )
        self.ent_linking_h2h_fc = build_classifier(
            input_size=decoder_hidden_size,
            output_size=3,
        )
        self.ent_linking_t2t_fc = build_classifier(
            input_size=decoder_hidden_size,
            output_size=3,
        )
        self.line_grouping_h2h_fc = build_classifier(
            input_size=decoder_hidden_size,
            output_size=3,
        )
        self.line_grouping_t2t_fc = build_classifier(
            input_size=decoder_hidden_size,
            output_size=3,
        )

        self.loss_ratio = config.peneo_loss_ratio
        if self.loss_ratio is not None:
            assert len(self.loss_ratio) == 5, "loss_ratio must be a list of 5 elements"
        category_weights = config.peneo_category_weights
        if category_weights is not None:
            assert (
                len(category_weights) == 3
            ), "category_weights must be a list of 3 elements"
            link_category_weights = torch.tensor(category_weights).float()
            le_category_weights = torch.tensor(category_weights[:-1]).float()
        self.link_loss = CrossEntropyLossOHEM(
            num_hard_positive=config.peneo_ohem_num_positive,
            num_hard_negative=config.peneo_ohem_num_negative,
            weight=link_category_weights,
        )
        self.le_loss = CrossEntropyLossOHEM(
            num_hard_positive=config.peneo_ohem_num_positive,
            num_hard_negative=config.peneo_ohem_num_negative,
            weight=le_category_weights,
        )

    def calculate_peneo_loss(
        self, pred: torch.Tensor, target: torch.Tensor, type: str = "link"
    ) -> torch.Tensor:
        """Calculate the loss for the PEneo sub-tasks

        Parameters
        ----------
        pred : torch.Tensor
            predicted tensor
        target : torch.Tensor
            ground truth tensor
        type : str, optional
            loss type, can be "link" or "le", by default "link"
        """
        assert len(pred.shape) == len(target.shape) + 1, f"invalid input shape"

        assert pred.shape[:-1] == target.shape
        pred_c = pred.shape[-1]
        if type == "link":
            return self.link_loss(pred.view(-1, pred_c), target.view(-1))
        else:
            return self.le_loss(pred.view(-1, pred_c), target.view(-1))

    def forward(
        self,
        sequence_output: torch.Tensor,
        orig_bbox: torch.Tensor = None,
        line_extraction_shaking_tag=None,
        ent_linking_head_rel_shaking_tag=None,
        ent_linking_tail_rel_shaking_tag=None,
        line_grouping_head_rel_shaking_tag=None,
        line_grouping_tail_rel_shaking_tag=None,
        **kwargs,
    ) -> PEneoOutput:
        if self.decoder_shrink:
            sequence_output = self.shrink_projection(sequence_output)
        shaking_hiddens = self.handshaking_kernel(sequence_output)
        shaking_hiddens4ent = shaking_hiddens
        shaking_hiddens4rel = shaking_hiddens

        line_extraction_shaking_outputs = self.line_extraction_fc(shaking_hiddens4ent)
        line_grouping_h2h_shaking_outputs = self.line_grouping_h2h_fc(
            shaking_hiddens4rel
        )
        line_grouping_t2t_shaking_outputs = self.line_grouping_t2t_fc(
            shaking_hiddens4rel
        )
        ent_linking_h2h_shaking_outputs = self.ent_linking_h2h_fc(shaking_hiddens4rel)
        ent_linking_t2t_shaking_outputs = self.ent_linking_t2t_fc(shaking_hiddens4rel)

        line_extraction_shaking_loss = self.calculate_peneo_loss(
            line_extraction_shaking_outputs, line_extraction_shaking_tag, type="le"
        )
        ent_linking_h2h_shaking_loss = self.calculate_peneo_loss(
            ent_linking_h2h_shaking_outputs,
            ent_linking_head_rel_shaking_tag,
            type="link",
        )
        ent_linking_t2t_shaking_loss = self.calculate_peneo_loss(
            ent_linking_t2t_shaking_outputs,
            ent_linking_tail_rel_shaking_tag,
            type="link",
        )
        line_grouping_h2h_shaking_loss = self.calculate_peneo_loss(
            line_grouping_h2h_shaking_outputs,
            line_grouping_head_rel_shaking_tag,
            type="link",
        )
        line_grouping_t2t_shaking_loss = self.calculate_peneo_loss(
            line_grouping_t2t_shaking_outputs,
            line_grouping_tail_rel_shaking_tag,
            type="link",
        )

        if self.loss_ratio is None:
            (
                loss_ratio_line_extraction,
                loss_ratio_ent_linking_h2h,
                loss_ratio_ent_linking_t2t,
                loss_ratio_line_grouping_h2h,
                loss_ratio_line_grouping_t2t,
            ) = (
                1,
                1,
                1,
                1,
                1,
            )
        else:
            (
                loss_ratio_line_extraction,
                loss_ratio_ent_linking_h2h,
                loss_ratio_ent_linking_t2t,
                loss_ratio_line_grouping_h2h,
                loss_ratio_line_grouping_t2t,
            ) = self.loss_ratio

        loss = (
            loss_ratio_line_extraction * line_extraction_shaking_loss
            + loss_ratio_ent_linking_h2h * ent_linking_h2h_shaking_loss
            + loss_ratio_ent_linking_t2t * ent_linking_t2t_shaking_loss
            + loss_ratio_line_grouping_h2h * line_grouping_h2h_shaking_loss
            + loss_ratio_line_grouping_t2t * line_grouping_t2t_shaking_loss
        )

        return PEneoOutput(
            loss=loss,
            line_extraction_loss=line_extraction_shaking_loss,
            ent_linking_h2h_loss=ent_linking_h2h_shaking_loss,
            ent_linking_t2t_loss=ent_linking_t2t_shaking_loss,
            line_grouping_h2h_loss=line_grouping_h2h_shaking_loss,
            line_grouping_t2t_loss=line_grouping_t2t_shaking_loss,
            line_extraction_shaking_outputs=line_extraction_shaking_outputs,
            ent_linking_h2h_shaking_outputs=ent_linking_h2h_shaking_outputs,
            ent_linking_t2t_shaking_outputs=ent_linking_t2t_shaking_outputs,
            line_grouping_h2h_shaking_outputs=line_grouping_h2h_shaking_outputs,
            line_grouping_t2t_shaking_outputs=line_grouping_t2t_shaking_outputs,
            orig_bbox=orig_bbox,
        )
