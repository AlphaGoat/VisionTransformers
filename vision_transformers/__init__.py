import torch
from typing import Tuple, Callable

from .detr import DETRBase
from .deformable_detr import DeformableDETRBase
from .loss import DETRLoss #, DeformableDETRLoss
from .dat import DeformableAttentionTransformerClassifier
from .backbones import build_vision_transformer_backbone, build_deformable_detr_backbone


def build_model(name: str='detr', backbone: str="resnet50", **kwargs) -> Tuple[torch.nn.Module, Callable]:
    """ Factory function to build vision transformer models.

    Args:
        name (str): Name of the model to build.
        **kwargs: Additional keyword arguments specific to the model.

    Returns:
        torch.nn.Module: Instantiated model.
    """
    if name == 'detr':
        model =  DETRBase(
            backbone=build_vision_transformer_backbone(
                backbone, 
                pretrained=kwargs.get('pretrained', True), 
                train_backbone=kwargs.get('train_backbone', False)
            ),
            num_classes=kwargs.get('num_classes', 91),
            num_queries=kwargs.get('num_queries', 100)
        )
        loss = DETRLoss(
            batch_size=kwargs.get('batch_size', 1),
            num_classes=kwargs.get('num_classes', 91),
            class_weight=kwargs.get('class_weight', 1.0),
            giou_weight=kwargs.get('giou_weight', 1.0),
            bbox_weight=kwargs.get('bbox_weight', 1.0)
        )
    elif name == 'deformable_detr':
        model = DeformableDETRBase(
            backbone=build_deformable_detr_backbone(
                backbone,
                pretrained=kwargs.get('pretrained', True), 
                train_backbone=kwargs.get('train_backbone', False)
            ),
            num_classes=kwargs.get('num_classes', 91),
            num_queries=kwargs.get('num_queries', 100)
        )
        loss = None # Todo: Implement DeformableDETRLoss
#        loss = DeformableDETRLoss(
#            num_classes=kwargs.get('num_classes', 91),
#            eos_coef=0.1,
#            losses=['labels', 'boxes']
#        )
    else:
        raise ValueError(f"Model '{name}' not recognized.")

    return model, loss