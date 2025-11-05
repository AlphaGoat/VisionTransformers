import torch
from detr import DETRBase
from deformable_detr import DeformableDETR
from loss import DETRLoss, DeformableDETRLoss
from dat import DeformableAttentionTransformerClassifier
from backbones import build_detr_backbone, build_deformable_detr_backbone
from typing import Tuple, Callable


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
            backbone=build_detr_backbone(backbone), 
            num_classes=kwargs.get('num_classes', 91),
            num_queries=kwargs.get('num_queries', 100)
        )
        loss = DETRLoss(
            num_classes=kwargs.get('num_classes', 91),
            eos_coef=0.1,
            losses=['labels', 'boxes']
        )
    elif name == 'deformable_detr':
        model = DeformableDETR(
            backbone=build_deformable_detr_backbone(backbone), 
            num_classes=kwargs.get('num_classes', 91),
            num_queries=kwargs.get('num_queries', 100)
        )
        loss = DeformableDETRLoss(
            num_classes=kwargs.get('num_classes', 91),
            eos_coef=0.1,
            losses=['labels', 'boxes']
        )
    else:
        raise ValueError(f"Model '{name}' not recognized.")

    return model, loss