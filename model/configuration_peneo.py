from typing import List

from transformers import PretrainedConfig


class PEneoConfig(PretrainedConfig):
    model_type = "peneo"

    def __init__(
        self,
        backbone_name: str = None,
        backbone_config: PretrainedConfig = None,
        initializer_range: float = 0.02,
        peneo_decoder_shrink: bool = True,
        peneo_classifier_num_layers: int = 2,
        peneo_loss_ratio: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        peneo_category_weights: List[float] = [1.0, 1.0, 1.0],
        peneo_ohem_num_positive: int = -1,
        peneo_ohem_num_negative: int = -1,
        peneo_downstream_speedup_ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.backbone_config = backbone_config
        self.initializer_range = initializer_range

        self.peneo_decoder_shrink = peneo_decoder_shrink
        self.peneo_classifier_num_layers = peneo_classifier_num_layers
        self.peneo_category_weights = peneo_category_weights
        self.peneo_loss_ratio = peneo_loss_ratio
        self.peneo_ohem_num_positive = peneo_ohem_num_positive
        self.peneo_ohem_num_negative = peneo_ohem_num_negative
        self.peneo_downstream_speedup_ratio = peneo_downstream_speedup_ratio
