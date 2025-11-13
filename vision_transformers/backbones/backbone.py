"""
Module for importing different backbones for Deformable DETR.

Author: Peter Thomas
Date: 07 October 2025
"""
from torchvision.models import resnet50, resnet101, vgg16
from torchvision.models.feature_extraction import create_feature_extractor
from vision_transformers.backbones.resnet_3d_conv import Resnet50With3dConv
from vision_transformers.utils import initialize_parameters


def fetch_resnet50_with_def_detr_hooks(pretrained=True):
    model = resnet50(pretrained=pretrained)
    return_nodes = {
        'layer4.2.relu_2': "feature3",
        'layer3.5.relu_2': "feature2",
        'layer2.3.relu_2': "feature1",
    }
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    return feature_extractor


def build_deformable_detr_backbone(name='resnet50', pretrained=True, train_backbone=False):
    if name == 'resnet50':
        backbone = fetch_resnet50_with_def_detr_hooks(pretrained=pretrained)
    elif name == 'resnet101':
#        backbone = resnet101(pretrained=pretrained)
        raise NotImplementedError("ResNet101 backbone is not implemented yet.")
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    for param in backbone.parameters():
        param.requires_grad = train_backbone

    return backbone


def build_vision_transformer_backbone(name='resnet50', pretrained=True, train_backbone=False):
    if name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        return_nodes = {
            "layer4": "feature_map",
        }
        backbone = create_feature_extractor(model, return_nodes=return_nodes)
    elif name == 'resnet101':
        backbone = resnet101(pretrained=pretrained)
    elif name == 'resnet50_with_3dconv':
        backbone = Resnet50With3dConv(classification_head=False)
    elif name == "vgg16":
        backbone = vgg16(pretrained=pretrained).features
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    for param in backbone.parameters():
        param.requires_grad = train_backbone

    if train_backbone and not pretrained:
        initialize_parameters(backbone, reset_backbone=True)

    return backbone