# Exercise Set 10 — Attention, Masking, and a Tiny Transformer Block

## Goal
This set introduces attention-related PyTorch tasks:
- scaled dot-product attention,
- attention masks,
- layer normalization,
- residual connections,
- a small Transformer-style block.

## Tasks
1. Implement `scaled_dot_product_attention`.
2. Implement `make_causal_mask`.
3. Implement `TinyTransformerBlock`.
4. Implement `masked_cross_entropy` that ignores padding targets.

## Hints
- Attention scores are `Q K^T / sqrt(d)`.
- For masked-out positions, fill attention scores with a large negative number before softmax.
- `nn.LayerNorm` works over the last dimension.
- Padding targets can be ignored via `ignore_index` in `cross_entropy`.

## Suggested References
- MultiheadAttention docs: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- Cross entropy: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
