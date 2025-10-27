"""
Module for importing different backbones for Deformable DETR.

Author: Peter Thomas
Date: 07 October 2025
"""
import torch
from torchvision.models import resnet50, resnet101
from torchvision.models.feature_extraction import create_feature_extractor


def fetch_resnet50(pretrained=True):
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
        backbone = fetch_resnet50(pretrained=pretrained)
    elif name == 'resnet101':
#        backbone = resnet101(pretrained=pretrained)
        raise NotImplementedError("ResNet101 backbone is not implemented yet.")
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    for param in backbone.parameters():
        param.requires_grad = train_backbone

    return backbone


def build_detr_backbone(name='resnet50', pretrained=True, train_backbone=False):
    if name == 'resnet50':
        backbone = resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = resnet101(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    for param in backbone.parameters():
        param.requires_grad = train_backbone

    return backbone