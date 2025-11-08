"""
Deformable Attention Model for classification problems

Author: Peter Thomas
Date: 07 October 2025
"""
import torch
#import deformable_attn_cuda
from torch.autograd import Function
from collections import OrderedDict

from .layers import SinusoidalPositionalEncoding
from .utils import get_num_output_channels


class DeformableAttention(Function):
    @staticmethod
    def forward(ctx, feature_map, sampling_points, queries, W_k, W_v, nhead):
        # Save inputs for backward pass
        ctx.save_for_backward(feature_map, sampling_points, queries, W_k, W_v)
        ctx.nhead = nhead

        # Call the CUDA kernel for forward pass
        output = deformable_attn_cuda.deformable_attn_forward(
            feature_map, sampling_points, queries, W_k, W_v, nhead
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_map, sampling_points, queries, W_k, W_v = ctx.saved_tensors
        nhead = ctx.nhead

        # Call the CUDA kernel for backward pass
        grad_queries, grad_keys, grad_values, grad_offsets, grad_feature_map = deformable_attn_cuda.deformable_attn_backward(
            grad_output, feature_map, sampling_points, queries, W_k, W_v, nhead
        )

        return grad_feature_map, grad_offsets, grad_queries, grad_keys, grad_values


class DeformableAttentionModule(torch.nn.Module):
    def __init__(self, d_model, nhead, downsample_factor=4, offset_scale=4):
        super(DeformableAttentionModule, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.downsample_factor = downsample_factor
        self.offset_scale = offset_scale

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)

        # Initialize weight matrices for deformable attention
        torch.nn.init.xavier_uniform_(self.W_q.weight)
        torch.nn.init.xavier_uniform_(self.W_k.weight)
        torch.nn.init.xavier_uniform_(self.W_v.weight)

        self.W_k = self.W_k.weight
        self.W_v = self.W_v.weight

        # Offset network for deformable attention
        self.offset_network = torch.nn.Sequential(
            OrderedDict([
                ("offset_depthwise_conv", torch.nn.Conv2d(d_model, d_model // self.downsample_factor, 
                                                          kernel_size=5, groups=self.downsample_factor)),
                ("offset_layer_norm", torch.nn.LayerNorm(d_model)),
                ("offset_gelu", torch.nn.GELU()),
                ("offset_conv", torch.nn.Conv2d(d_model, nhead * 2, kernel_size=1))
            ])
        )

        # Deformable attention method
        self.deformable_attention = DeformableAttention.apply

    def forward(self, feature_map):

        # Get reference points
        reference_points = self.calculate_reference_points(feature_map)

        # Reshape features for attention
        features = feature_map.view(feature_map.size(0), feature_map.size(1), -1)  # (batch_size, C, H*W)
        features = feature_map.permute(0, 2, 1)  # (batch_size, H*W, C)

        # Generate queries
        queries = self.W_q(features)

        # Get offsets from offset network
        offsets = self.offset_network(features)
        offsets *= self.offset_scale

        # Add offsets to reference points
        sampling_points = reference_points.unsqueeze(1) + offsets.view(feature_map.size(0), -1, self.nhead, 2)

        # Call the CUDA kernel for deformable attention

        attn_output = self.deformable_attention(
            feature_map, sampling_points, queries, self.W_k, self.W_v, self.nhead
        )

        return attn_output

    def calculate_reference_points(self, feature_map):
        B, C, H, W = feature_map.size()
        device = feature_map.device

        # Create a mesh grid of normalized coordinates
        y_coords = torch.linspace(-1, 1, H // self.downsample_factor, device=device)
        x_coords = torch.linspace(-1, 1, W // self.downsample_factor, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        y_grid = y_grid.reshape(-1)
        x_grid = x_grid.reshape(-1)

        reference_points = torch.stack((x_grid, y_grid), dim=-1)  # Shape: (H*W, 2)
        reference_points = reference_points.unsqueeze(0).repeat(B, 1, 1)  # Shape: (B, H*W, 2)

        return reference_points


class DeformableAttentionTransformerClassifier(torch.nn.Module):
    def __init__(self, backbone, image_shape=(3, 256, 256), nhead=8, num_classes=100, position_encoding="sinusoidal"):
        super(DeformableAttentionTransformerClassifier, self).__init__()
        self.backbone = backbone
        self.nhead = nhead

        # The dimension of the model is the number of channels output by the backbone
        self.d_model = get_num_output_channels(backbone, image_shape)

        # Assert that d_model is divisible by nhead
        assert self.d_model % nhead == 0, "d_model must be divisible by nhead"

        # Positional Encoding
        if position_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(self.d_model)
        else:
            raise ValueError("Unknown positional encoding type")

        # Deformable Attention Layer
        self.deformable_attention = DeformableAttention(self.d_model, nhead)

        # Classifier head
        self.classifier = torch.nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        # Extract features using the backbone
        features = self.backbone(x)  # Assume features shape: (batch_size, C, H, W)

        attn_output, _ = self.deformable_attention(features)

        # Take the mean across the sequence length dimension
        pooled_output = attn_output.mean(dim=0)
        logits = self.classifier(pooled_output)
        return logits