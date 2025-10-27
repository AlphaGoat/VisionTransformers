from detr import DETRBase
from loss import DETRLoss
from backbone import build_detr_backbone


def build_model(name='detr', backbone="resnet50",**kwargs):
    """ Factory function to build vision transformer models.

    Args:
        name (str): Name of the model to build.
        **kwargs: Additional keyword arguments specific to the model.

    Returns:
        torch.nn.Module: Instantiated model.
    """
    # Build backbone

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
    else:
        raise ValueError(f"Model '{name}' not recognized.")