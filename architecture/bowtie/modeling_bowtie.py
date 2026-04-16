import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.configuration_utils import PretrainedConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


class ResidualAdapter(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * self.proj(x)


class BowtieAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, eps: float = 1e-6):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps)
        self.norm2 = RMSNorm(dim, eps)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        if attention_mask is not None:
            causal_mask = causal_mask | attention_mask.unsqueeze(1).unsqueeze(2)

        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class BowtieConfig(PretrainedConfig):
    model_type = "bowtie"

    def __init__(
        self,
        vocab_size=50257,
        d_model=512,
        d_small=85,
        n_layers=8,
        n_heads=8,
        max_position_embeddings=128,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_small = d_small
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps


class BowtiePreTrainedModel(PreTrainedModel):
    config_class = BowtieConfig
    base_model_prefix = "bowtie"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BowtieAttentionBlock", "ResidualAdapter"]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.scale.data.fill_(1.0)


class BowtieModel(BowtiePreTrainedModel):
    def __init__(self, config: BowtieConfig):
        super().__init__(config)
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(1, config.max_position_embeddings, config.d_model) * config.initializer_range
        )

        self.layer_1 = BowtieAttentionBlock(config.d_model, config.n_heads, config.rms_norm_eps)
        self.down_proj = nn.Linear(config.d_model, config.d_small, bias=False)
        self.entry_skip = ResidualAdapter(config.d_model, config.d_small)

        n_mid = config.n_layers - 2
        heads_small = config.n_heads if config.d_small % config.n_heads == 0 else 1
        self.middle_layers = nn.ModuleList([
            BowtieAttentionBlock(config.d_small, heads_small, config.rms_norm_eps)
            for _ in range(n_mid)
        ])

        self.up_proj = nn.Linear(config.d_small, config.d_model, bias=False)
        self.global_skip = ResidualAdapter(config.d_model, config.d_model)
        self.exit_skip = ResidualAdapter(config.d_small, config.d_model)
        self.layer_L = BowtieAttentionBlock(config.d_model, config.n_heads, config.rms_norm_eps)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        x = self.embed(input_ids) + self.pos_emb[:, :input_ids.size(1), :]

        h1 = self.layer_1(x, attention_mask)
        h_small = self.down_proj(h1) + self.entry_skip(h1)

        for layer in self.middle_layers:
            h_small = layer(h_small, attention_mask)

        h_big = self.up_proj(h_small) + self.global_skip(h1) + self.exit_skip(h_small)
        output = self.layer_L(h_big, attention_mask)

        return output


class BowtieForCausalLM(BowtiePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: BowtieConfig):
        super().__init__(config)
        self.bowtie = BowtieModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.bowtie.embed

    def set_input_embeddings(self, value: nn.Embedding):
        self.bowtie.embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.bowtie(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        if not return_dict:
            return (logits, loss) if loss is not None else (logits,)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
