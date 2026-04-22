# Bowtie Transformer

> A Transformer architecture with bottleneck layers and multi-level residual connections.

```
d_model ──[Big Layer 1]──► BottleneckProjection↓ ──► [Small Layers ×24] ──► BottleneckProjection↑ ──[Big Layer 26]──► output
    │                                                                                                             ▲
    └────────────────────────────── GlobalSkip (ResidualAdapter) ─────────────────────────────────────────────────┘
```

## Overview

Standard Transformers allocate equal compute across all layers, regardless of semantic importance. **Bowtie Transformer** challenges this: the first and last layers operate at full dimension `d_model=512`, while 24 intermediate layers work in a compressed space `d_small=128` (4× reduction). The compute graph resembles a bowtie: **wide → narrow → wide**.

This design concentrates representational capacity at boundary layers while using compressed intermediate layers as a soft regularizer. Three long-range residual paths with zero-initialized adapters prevent information loss across dimension boundaries. In practice, Bowtie achieves **17.8% fewer parameters** and **lower perplexity** than a standard transformer of comparable capacity.

## Key Ideas

**Asymmetric layer dimensions**
- Layers 1 & 26: full `d_model = 512`
- Layers 2…25: compressed `d_small = 128` (`compression_ratio = 4`)

**BottleneckProjection** — independent learned projections at each dimension boundary, stabilized by RMSNorm. Down- and up-projection weights are not tied, allowing asymmetric feature compression and reconstruction.

**Three residual paths with ResidualAdapter**

| Path | From → To | Projection |
|---|---|---|
| `GlobalSkip` | Layer 1 → Layer 26 | `d_model → d_model` |
| `EntrySkip` | Layer 1 → first small layer | `d_model → d_small` |
| `ExitSkip` | last small layer → Layer 26 | `d_small → d_model` |

Each `ResidualAdapter` uses a learnable scale `γ` initialized to **zero**. The extra paths are effectively disabled at initialization and open gradually during training (zero-init residual strategy), ensuring stable gradients in deep networks.

## Results

Trained for 1500 steps on `roneneldan/TinyStories` (seq_len=128, batch=32, lr=5e-4, AMP). All models optimized to comparable parameter budgets.

| Model | Params | Layers | Final Loss | Perplexity 🎯 | Δ PPL vs Standard |
|---|---|---|---|---|---|
| Standard | 76.74M | 8 | 3.1563 | 23.48 | — |
| **Bowtie** | **63.11M** | **26 (2+24)** | **3.1246** | **22.75** | **▼ 3.1%** |

- **17.8% parameter reduction** (~13.6M fewer weights)
- **3.1% lower perplexity** at equal training budget
- Stable convergence despite 26-layer depth (PreNorm + residual adapters)
- Effective depth ≈ 3.5 full-width layers due to compression

## Architecture

```python
# Forward pass overview (PyTorch-style pseudocode)
h1      = BigLayer_1(x)

h_small = DownProjection(h1)
h_small += ResidualAdapter_entry(h1)          # EntrySkip: d_model → d_small

for i in range(24):
    h_small = SmallLayer_i(h_small)

h_big   = UpProjection(h_small)
h_big  += ResidualAdapter_global(h1)          # GlobalSkip: d_model → d_model
h_big  += ResidualAdapter_exit(h_small)        # ExitSkip: d_small → d_model

output  = BigLayer_26(h_big)
```

## Configuration

```python
d_model        = 512
d_small        = 128   # compression_ratio = 4 (d_model / d_small)
n_layers       = 26    # 2 big + 24 small
n_heads        = 8     # divisible by d_small for multi-head attention
max_seq_len    = 128
```

> 💡 **Analytical tuning**: Use `compression_ratio ∈ {4, 6, 8}` to balance parameter savings vs. representational capacity. `ratio=4` proved optimal for TinyStories.

## References

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- He et al. (2016) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Zhang & Sennrich (2019) — [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- Liu et al. (2022) — [A ConvNet for the 2020s (zero-init residual)](https://arxiv.org/abs/2201.03545)
- Su et al. (2021) — [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Ainslie et al. (2023) — [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- Shazeer (2020) — [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
