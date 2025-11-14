"""
Deformable Attention Model for classification problems

Author: Peter Thomas
Date: 07 October 2025
"""
import torch
#import deformable_attn_cuda
from torch.autograd import Function
from collections import OrderedDict

from .layers import SinusoidalPositionalEncoding, MultiHeadAttention, ShiftedWindowAttention
from .utils import get_num_output_channels, initialize_parameters
from .cuda import deformable_attn_cuda


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

        # Call CUDA kernel for bilinear interpolation of sampling points
        sampled_features = deformable_attn_cuda.bilinear_interpolate(feature_map, sampling_points)

        # Generate keys and values
        keys = torch.matmul(sampled_features, self.W_k)
        values = torch.matmul(sampled_features, self.W_v)

        # Call multihead attention
        attn_output = self.multihead_attention(
            queries, keys, values, reference_points, self.nhead
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


class ShiftedWindowAttentionStage(torch.nn.Module):
    def __init__(self, d_model, nhead, window_size=7, shift_size=3):
        super(ShiftedWindowAttentionStage, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.shift_size = shift_size

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.local_attention = MultiHeadAttention(d_model, nhead)
        self.shifted_window_attention = ShiftedWindowAttention(d_model, nhead, window_size, shift_size)

    def forward(self, x):
        # First local attention on patch inputs
        local_attn_output = self.local_attention(x, x, x)

        # Then shifted window attention
        attn_output = self.shifted_window_attention(local_attn_output)
        return attn_output


class DeformableAttentionStage(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super(DeformableAttentionStage, self).__init__()
        self.d_model = d_model
        self.nhead = nhead

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.local_attention = MultiHeadAttention(d_model, nhead)
        self.deformable_attention = DeformableAttentionModule(d_model, nhead)

    def forward(self, x):
        # First local attention on patch inputs
        local_attn_output = self.local_attention(x, x, x)

        # Then deformable attention
        attn_output = self.deformable_attention(local_attn_output)
        return attn_output


class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(PatchEmbedding, self).__init__()
        self.proj = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.activation = torch.nn.GELU()
        self.norm = torch.nn.LayerNorm(out_channels)

        # Initialize projection weights
        torch.nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        x = self.proj(x)  # (B, out_channels, H', W')
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.norm(x)
        return x


class DeformableAttentionTransformerClassifier(torch.nn.Module):
    def __init__(self, d_model, image_shape=(3, 256, 256), 
                 nhead=8, num_classes=100, position_encoding="sinusoidal"):
        super(DeformableAttentionTransformerClassifier, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_classes = num_classes

        # Assert that d_model is divisible by nhead
        assert self.d_model % nhead == 0, "d_model must be divisible by nhead"

        # Positional Encoding
        if position_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(self.d_model)
        else:
            raise ValueError("Unknown positional encoding type")

        self.patch_embedding_1 = PatchEmbedding(
            in_channels=image_shape[0],
            out_channels=self.d_model,
            kernel_size=4,
            stride=4
        )

        self.patch_embedding_2 = PatchEmbedding(
            in_channels=self.d_model,
            out_channels=2 * self.d_model,
            kernel_size=2,
            stride=2
        )

        self.patch_embedding_3 = PatchEmbedding(
            in_channels=2 * self.d_model,
            out_channels=4 * self.d_model,
            kernel_size=2,
            stride=2
        )

        self.patch_embedding_4 = PatchEmbedding(
            in_channels=4 * self.d_model,
            out_channels=8 * self.d_model,
            kernel_size=2,
            stride=2
        )

        # Different stages of DAT architecture
        self.stage1 = ShiftedWindowAttentionStage(d_model=self.d_model, nhead=self.nhead, window_size=7, shift_size=3)
        self.stage2 = ShiftedWindowAttentionStage(d_model=2 * self.d_model, nhead=self.nhead, window_size=7, shift_size=3)
        self.stage3 = DeformableAttentionStage(d_model=4 * self.d_model, nhead=self.nhead)
        self.stage4 = DeformableAttentionStage(d_model=8 * self.d_model, nhead=self.nhead)

        # Classifier head
        self.classifier = torch.nn.Linear(8 * self.d_model, num_classes)
        self.final_activation = torch.nn.Softmax(dim=-1)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        # Generate image patches
        patches = self.generate_image_patches(x)

        # Run through different stages of DAT
        x = self.patch_embedding_1(patches)
        x = self.stage1(x)
        x = self.patch_embedding_2(x)
        x = self.stage2(x)
        x = self.patch_embedding_3(x)
        x = self.stage3(x)
        x = self.patch_embedding_4(x)
        x = self.stage4(x)

        # Take the mean across the sequence length dimension
        logits = self.classifier(x)
        return self.final_activation(logits)
