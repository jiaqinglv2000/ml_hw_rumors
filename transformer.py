import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [bsz, seq_len, n_head, head_size]
        return x.permute(0, 2, 1, 3)  # [bsz, n_head, seq_len, head_size]

    def forward(self, q, k, v, attention_mask=None):
        query = self.transpose_for_scores(self.query(q))  # [bsz, n_head, lq, head_size]
        key = self.transpose_for_scores(self.key(k))  # [bsz, n_head, lk, head_size]
        value = self.transpose_for_scores(self.value(v))  # [bsz, n_head, lv, head_size]

        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [bsz, n_head, lq, lk]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]  # [bsz, 1, 1, lk]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]  # [bsz, 1, lq, lk]
            attention_scores = attention_scores.masked_fill(attention_mask, -1e9)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)  # [bsz, n_head, lq, lk]

        context = torch.matmul(attention_probs, value)  # [bsz, n_head, lq, head_size]
        context = context.permute(0, 2, 1, 3).contiguous()  # [bsz, lq, n_head, head_size]

        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)  # [bsz, lq, dim_hidden]

        return context, attention_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_dropout, dropout):
        super().__init__()
        self.SDPA = ScaledDotProductAttention(hidden_size, num_heads, attn_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        q, k, v = hidden_states, hidden_states, hidden_states
        context, attention_probs = self.SDPA(q, k, v, attention_mask)
        context = self.dense(context)
        context = self.dropout(context)

        hidden_states = self.LayerNorm(hidden_states + context)
        return hidden_states, attention_probs


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.LN = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.LN(x + self.net(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attn_dropout=.0, dropout=.0):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_heads, attn_dropout, dropout)
        self.ffn = FeedForwardNetwork(hidden_size, intermediate_size, dropout)

    def forward(self, hidden_states, attention_mask):
        hidden_states, _ = self.mha(hidden_states, attention_mask)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, intermediate_size, attn_dropout=.0, dropout=.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size, attn_dropout, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x
import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)  # [bsz, seq_len, n_head, head_size]
        return x.permute(0, 2, 1, 3)  # [bsz, n_head, seq_len, head_size]

    def forward(self, q, k, v, attention_mask=None):
        query = self.transpose_for_scores(self.query(q))  # [bsz, n_head, lq, head_size]
        key = self.transpose_for_scores(self.key(k))  # [bsz, n_head, lk, head_size]
        value = self.transpose_for_scores(self.value(v))  # [bsz, n_head, lv, head_size]

        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # [bsz, n_head, lq, lk]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]  # [bsz, 1, 1, lk]
            if attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]  # [bsz, 1, lq, lk]
            attention_scores = attention_scores.masked_fill(attention_mask, -1e9)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)  # [bsz, n_head, lq, lk]

        context = torch.matmul(attention_probs, value)  # [bsz, n_head, lq, head_size]
        context = context.permute(0, 2, 1, 3).contiguous()  # [bsz, lq, n_head, head_size]

        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)  # [bsz, lq, dim_hidden]

        return context, attention_probs


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, attn_dropout, dropout):
        super().__init__()
        self.SDPA = ScaledDotProductAttention(hidden_size, num_heads, attn_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        q, k, v = hidden_states, hidden_states, hidden_states
        context, attention_probs = self.SDPA(q, k, v, attention_mask)
        context = self.dense(context)
        context = self.dropout(context)

        hidden_states = self.LayerNorm(hidden_states + context)
        return hidden_states, attention_probs


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.LN = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.LN(x + self.net(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, attn_dropout=.0, dropout=.0):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_heads, attn_dropout, dropout)
        self.ffn = FeedForwardNetwork(hidden_size, intermediate_size, dropout)

    def forward(self, hidden_states, attention_mask):
        hidden_states, _ = self.mha(hidden_states, attention_mask)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, intermediate_size, attn_dropout=.0, dropout=.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size, attn_dropout, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, attn_mask):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

