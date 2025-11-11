import torch


class AttentionLayer(torch.nn.Module):
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
        self.attention_layers = torch.nn.ModuleList([AttentionLayer(d_model // nhead, nhead) for _ in range(nhead)])
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

