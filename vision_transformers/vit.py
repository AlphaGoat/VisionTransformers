"""
My implementation of ViT (Vision Transformer) model.

Author: Peter Thomas
Date: 28 October 2025
"""
import torch
from layers import MultiHeadAttention, SinusoidalPositionalEncoding


class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        """
        Transformer Encoder Block.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            mlp_ratio (float): Ratio of the hidden dimension in the MLP to embed_dim
        """
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = torch.nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        # Multi-head attention with residual connection
        x = x + self.mha(self.norm1(x), self.norm1(x), self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(torch.nn.Module):
    def __init__(self, img_size, num_channels, patch_size, num_classes, embed_dim, depth, num_heads, num_encoder_blocks, pos_encoding="sinusoidal", mlp_ratio=4.0):
        """
        Vision Transformer (ViT) model.

        Args:
            img_size (int): Size of the input image (assumed square).
            num_channels (int): Number of channels in the input image.
            patch_size (int): Size of each image patch (assumed square).
            num_classes (int): Number of output classes.
            embed_dim (int): Output dimension of the linear projection embedding. 
            num_heads (int): Number of attention heads in each MHA layer.
            num_encoder_blocks (int): Number of transformer encoder blocks.
            pos_encoding (str): Type of positional encoding to use ("sinusoidal" or "learned").
            mlp_ratio (float): Ratio of the hidden dimension in the MLP to embed_dim
        """
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.num_encoder_blocks = num_encoder_blocks

        # Implementation of Vision Transformer goes here
        self.linear_proj = torch.nn.Linear(patch_size * patch_size * num_channels, embed_dim)

        if pos_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(self.num_patches, embed_dim)
        elif pos_encoding == "learned":
            self.positional_encoding = torch.nn.Embedding(self.num_patches, embed_dim)
        else:
            raise ValueError("Unsupported positional encoding type.")

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_encoder_blocks)
        ])
        self.classification_head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        patches = self.generate_image_patches(x)

        # Forward pass implementation
        linear_projections = []
        for i in range(self.num_patches):
            # Learned linear projections for each patch
            patch = patches[:, :, i]

            # Flatten patch in spatial and channel dimensions
            patch = patch.flatten(start_dim=1)
            patch = self.linear_proj(patch)
            linear_projections.append(patch)
        encoder_in = linear_projections = torch.stack(linear_projections, dim=1)

        # Transformer encoder blocks
        for i in range(self.num_encoder_blocks):
            encoder_in = self.transformer_blocks[i](encoder_in)
        encoder_out = encoder_in

        # Final classification head
        logits = self.classification_head(encoder_out[:, 0, :])  # Use the representation of the [CLS] token
        return logits

    def generate_image_patches(self, x):
        # Method to generate image patches
        B, C, H, W = x.shape
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        return x