"""
Hook functions for model training and eval.

Author: Peter Thomas
Date: 28 October 2025
"""
import torch
from typing import Dict
from vision_transformers.layers import AttentionLayer


def get_attention_weights(model: torch.nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Logs the attention weights of the model's transformer layers.

    Args:
        model: The vision transformer model.
    Returns:
        A dictionary with layer names as keys and another dict containing keys,
        query, and value matrices as their values.
    """
    attention_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, AttentionLayer):
            attention_weights[name] = {
                "keys": module.W_k.weight.data.cpu(),
                "queries": module.W_q.weight.data.cpu(),
                "values": module.W_v.weight.data.cpu()
            }
    
    return attention_weights


def get_layer_statistics(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Computes statistics (mean and std) of the weights of each layer in the model.

    Args:
        model: The vision transformer model.

    Returns:
        A dictionary with layer names as keys and their weight statistics as values.
    """
    layer_stats = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            layer_stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item()
            }

    return layer_stats