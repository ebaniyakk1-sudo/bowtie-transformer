# Bowtie Transformer

> A Transformer architecture with bottleneck layers and multi-level residual connections.

```
d_model ──[Big Layer 1]──► BottleneckProjection↓ ──► [Small Layers...] ──► BottleneckProjection↑ ──[Big Layer L]──► output
    │                                                                                                        ▲
    └────────────────────────────── GlobalSkip (ResidualAdapter) ───────────────────────────────────────────┘
```

## Overview

Standard Transformers use the same hidden dimension across all layers — allocating equal compute regardless of semantic importance. **Bowtie Transformer** challenges this: boundary layers (first and last) operate at full dimension, while intermediate layers work in a compressed space **5–6× smaller**. The compute graph resembles a bowtie: wide → narrow → wide.

This concentrates model capacity where it matters most, while three types of long-range residual connections prevent information loss across dimension boundaries.

## Key Ideas

**Asymmetric layer dimensions**
- Layers 1 and L: full `d_model`
- Layers 2…L−1: compressed `d_small = d_model / k`, where `k ∈ {5, 6}`

**BottleneckProjection** — learned projections at each dimension boundary with RMSNorm for activation scale stability. Down- and up-projection weights are trained independently (no weight tying).

**Three residual paths with ResidualAdapter**

| Path | From → To | Projection |
|---|---|---|
| `GlobalSkip` | Layer 1 → Layer L | `d_model → d_model` |
| `EntrySkip` | Layer 1 → first small layer | `d_model → d_small` |
| `ExitSkip` | last small layer → Layer L | `d_small → d_model` |

Each `ResidualAdapter` uses a learnable scale vector `γ` initialized to **zero** — the extra paths are gated off at init and open gradually during training (zero-init residual).

## Results

Trained for 1500 steps, ~76M parameters, compared against Standard and Hybrid baselines:

| Model | Params | Final Loss | Train Time | Complexity Score |
|---|---|---|---|---|
| Standard | 76.7M | 3.3281 | ~345s | ~243 |
| **Bowtie** | **76.3M** | **3.1262** | ~375s | **~220** |

- **~7.5% lower loss** vs Standard at end of training
- Lower Complexity Score (Loss × Params) despite similar parameter count
- Bowtie leads from step ~100 onward; initial overhead recovers quickly

## Architecture

```
Input: x  (B, T, d_model)

h₁        = BigLayer_1(x)

h_small   = BottleneckProjection↓(h₁)
h_small  += ResidualAdapter_entry(h₁)          # EntrySkip

for i in 1…L−2:
    h_small = SmallLayer_i(h_small)

h_big     = BottleneckProjection↑(h_small)
h_big    += ResidualAdapter_global(h₁)          # GlobalSkip
h_big    += ResidualAdapter_exit(h_small)        # ExitSkip

output    = BigLayer_L(h_big)
```

## Configuration

```python
d_model        = 512
d_small        = d_model // 6   # ≈ 85  (or // 5 = 102)
num_big_layers = 2               # layers 1 and L
num_small_layers = L - 2
```

## References

- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- He et al. (2016) — [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- Zhang & Sennrich (2019) — [RMSNorm](https://arxiv.org/abs/1910.07467)
- Liu et al. (2022) — [ConvNeXt / zero-init residual](https://arxiv.org/abs/2201.03545)
- Su et al. (2024) — [RoPE](https://arxiv.org/abs/2104.09864)
- Ainslie et al. (2023) — [GQA](https://arxiv.org/abs/2305.13245)
- Shazeer (2020) — [GLU Variants](https://arxiv.org/abs/2002.05202)
