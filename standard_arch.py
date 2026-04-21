
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale

class PreNormBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn   = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
    def forward(self, x):
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), 1).bool()
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class ResidualAdapter(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.gamma * self.proj(x)

class StandardConfig(PretrainedConfig):
    model_type = "standard_transformer"
    def __init__(self, vocab_size=50257, d_model=512, n_layers=8, max_seq_len=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.d_model, self.n_layers, self.max_seq_len = vocab_size, d_model, n_layers, max_seq_len

class StandardTransformer(PreTrainedModel):
    config_class = StandardConfig
    def __init__(self, config):
        super().__init__(config)
        d = config.d_model
        self.embed = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, d) * 0.02)
        self.layers = nn.ModuleList([PreNormBlock(d, 8) for _ in range(config.n_layers)])
        self.head = nn.Linear(d, config.vocab_size, bias=False)
    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids) + self.pos_emb[:, :input_ids.size(1), :]
        for layer in self.layers: x = layer(x)
        return self.head(x)
