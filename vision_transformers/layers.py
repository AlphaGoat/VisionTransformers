"""
Module for common layers in models.

Author: Peter Thomas
Date: 07 October 2025
"""
import torch


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
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x