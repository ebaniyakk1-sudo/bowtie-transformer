
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

class BowtieConfig(PretrainedConfig):
    model_type = "bowtie_transformer"
    def __init__(self, vocab_size=50257, d_model=512, d_small=128, n_layers=26, max_seq_len=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.d_model, self.d_small, self.n_layers, self.max_seq_len = vocab_size, d_model, d_small, n_layers, max_seq_len

class BowtieTransformer(PreTrainedModel):
    config_class = BowtieConfig
    def __init__(self, config):
        super().__init__(config)
        d, s = config.d_model, config.d_small
        self.embed = nn.Embedding(config.vocab_size, d)
        self.pos_emb = nn.Parameter(torch.randn(1, config.max_seq_len, d) * 0.02)
        self.layer_1, self.down_proj, self.entry_skip = PreNormBlock(d, 8), nn.Linear(d, s), ResidualAdapter(d, s)
        self.middle_layers = nn.ModuleList([PreNormBlock(s, 8 if s%8==0 else 1) for _ in range(config.n_layers-2)])
        self.up_proj, self.global_skip, self.exit_skip = nn.Linear(s, d), ResidualAdapter(d, d), ResidualAdapter(s, d)
        self.layer_L, self.head = PreNormBlock(d, 8), nn.Linear(d, config.vocab_size, bias=False)
    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids) + self.pos_emb[:, :input_ids.size(1), :]
        h1 = self.layer_1(x)
        h_small = self.down_proj(h1) + self.entry_skip(h1)
        for layer in self.middle_layers: h_small = layer(h_small)
        h_big = self.up_proj(h_small) + self.global_skip(h1) + self.exit_skip(h_small)
        return self.head(self.layer_L(h_big))
