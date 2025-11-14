"""
Module for attention layers used in Vision Transformers.

Author: Peter Thomas
Date: 13 November 2025
"""
import torch


class ScaledDotProductAttentionLayer(torch.nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)

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
        self.attention_layers = torch.nn.ModuleList([ScaledDotProductAttentionLayer(d_model // nhead) for _ in range(nhead)])
        self.linear = torch.nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        # Divide queries, keys, values for each head
        query = query.view(query.size(0), query.size(1), self.nhead, -1).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.nhead, -1).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.nhead, -1).transpose(1, 2)

        attn_outputs = [attn_layer(q.squeeze(1), k.squeeze(1), v.squeeze(1)) for attn_layer, q, k, v in
                        zip(self.attention_layers, torch.split(query, 1, dim=1),
                            torch.split(key, 1, dim=1), torch.split(value, 1, dim=1))]
        concat_attn = torch.cat(attn_outputs, dim=-1)
        output = self.linear(concat_attn)
        return output


class ShiftedWindowAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, window_size=7, shift_size=3):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size
        self.shift_size = shift_size

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.attention = MultiHeadAttention(d_model, nhead)

    def forward(self, x):
        # Extract windows from input patches
        windows = self.extract_windows(x)
        attn_output = self.attention(windows, windows, windows)
        return attn_output

    def extract_windows(self, x):
        """
        Extract non-overlapping windows from the input patch embeddings.
        Args:
            x (torch.Tensor): Input patch embeddings of shape (B, H, W, C).
        Returns:
            torch.Tensor: Extracted windows of shape (num_windows*B, window_size*window_size, C).
        """
        (B, H, W, C) = x.size()
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return x

    def combine_windows(self, x, original_size):
        """
        Combine windows back to the original patch embedding shape.
        Args:
            x (torch.Tensor): Windows of shape (num_windows*B, window_size*window_size, C).
            original_size (tuple): Original size (B, H, W, C).
        Returns:
            torch.Tensor: Combined patch embeddings of shape (B, H, W, C).
        """
        (B, H, W, C) = original_size
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return x