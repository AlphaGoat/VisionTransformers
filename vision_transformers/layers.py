"""
Module for common layers in models.

Author: Peter Thomas
Date: 07 October 2025
"""
import torch
from vision_transformers.utils import get_x_positions, get_y_positions


class AttentionLayer(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        softmax = torch.nn.Softmax(dim=-1)

        attn_output = torch.bmm(softmax(torch.bmm(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)), V)
        return attn_output


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.attention_layers = torch.nn.ModuleList([AttentionLayer(d_model // nhead, nhead) for _ in range(nhead)])
        self.linear = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        attn_outputs = [attn_layer(query, key, value) for attn_layer in self.attention_layers]
        concat_attn = torch.cat(attn_outputs, dim=-1)
        output = self.linear(concat_attn)
        return output


class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, n_pixels, embed_dim):
        super().__init__()
        self.n_pixels = n_pixels
        self.embed_dim = embed_dim // 2 # Since we have x and y positions

        # get axis specific positions
        self.x_positions = get_x_positions(n_pixels)
        self.y_positions = get_y_positions(n_pixels)

        # Generate positional encodings
        x_pos_embedding = self.generate_sinusoidal_1d(self.x_positions.unsqueeze(1))
        y_pos_embedding = self.generate_sinusoidal_1d(self.y_positions.unsqueeze(1))

        # Combine x-axis and y-axis positional encodings
        pe = torch.cat((x_pos_embedding, y_pos_embedding), dim=-1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

    def generate_sinusoidal_1d(self, sequence):
        # Denominator
        denominator = torch.pow(10000, torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)

        # Create empty tensor for positional encodings
        pos_encodings = torch.zeros((sequence.size(0), self.embed_dim))
        denominator = sequence / denominator
        pos_encodings[:, :, 0::2] = torch.sin(denominator)
        pos_encodings[:, :, 1::2] = torch.cos(denominator)
        return pos_encodings