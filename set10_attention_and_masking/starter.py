import math
import unittest
import torch


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """
    Implement scaled dot-product attention.
    Shapes:
        q, k, v: (B, T, D)
        mask: (T, T) or (B, T, T), True where attention is allowed
    Return:
        output: (B, T, D)
        attn: (B, T, T)
    """
    # TODO: compute scaled attention scores, apply mask, softmax, and weighted sum.
    # Обчислюємо attention scores за формулою QK^T / sqrt(D).
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)

    # Якщо передано mask, заборонені позиції заповнюємо дуже малим значенням.
    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, -1e9)

    # Перетворюємо scores на ймовірності уваги.
    attn = torch.softmax(scores, dim=-1)

    # Обчислюємо зважену суму значень V.
    output = torch.matmul(attn, v)

    return output, attn


def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    Return a boolean causal mask of shape (T, T), where position i can attend to <= i.
    """
    # TODO: create a lower-triangular boolean mask.
    # Створюємо нижньотрикутну матрицю, де True дозволяє увагу.
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


class TinyTransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        """
           Build a tiny transformer-style block with:
            - q_proj, k_proj, v_proj
            - attention
            - residual + layernorm
            - feed-forward: Linear -> ReLU -> Linear
            - residual + layernorm
        """
        # TODO: define submodules.
        # Створюємо проєкції для Q, K та V.
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)

        # Створюємо проєкцію після attention.
        self.out_proj = torch.nn.Linear(d_model, d_model)

        # Створюємо LayerNorm для двох residual-блоків.
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

        # Створюємо feed-forward блок.
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(d_hidden, d_model)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO: implement the transformer block forward pass.
        # Обчислюємо Q, K та V для attention.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Виконуємо scaled dot-product attention.
        attn_out, _ = scaled_dot_product_attention(q, k, v, mask)

        # Застосовуємо вихідну проєкцію після attention.
        attn_out = self.out_proj(attn_out)

        # Перший residual connection + layer normalization.
        x = self.norm1(x + attn_out)

        # Feed-forward блок.
        ff_out = self.ff(x)

        # Другий residual connection + layer normalization.
        x = self.norm2(x + ff_out)

        return x


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    Compute token-level cross entropy while ignoring padding positions.
    logits: (B, T, C)
    targets: (B, T)
    """
    # TODO: flatten batch/time dimensions and call cross_entropy.
    # Розгортаємо logits з форми (B, T, C) у форму (B*T, C).
    num_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, num_classes)

    # Розгортаємо targets з форми (B, T) у форму (B*T).
    targets_flat = targets.reshape(-1)

    # Обчислюємо cross entropy, ігноруючи padding-позиції.
    return torch.nn.functional.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=ignore_index
    )


class TestAttentionAndTransformer(unittest.TestCase):
    def test_make_causal_mask(self):
        got = make_causal_mask(4)
        expected = torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ])
        self.assertTrue(torch.equal(got, expected))

    def test_attention_shapes(self):
        torch.manual_seed(0)
        q = torch.randn(2, 5, 8)
        k = torch.randn(2, 5, 8)
        v = torch.randn(2, 5, 8)
        out, attn = scaled_dot_product_attention(q, k, v)
        self.assertEqual(tuple(out.shape), (2, 5, 8))
        self.assertEqual(tuple(attn.shape), (2, 5, 5))
        self.assertTrue(torch.allclose(attn.sum(dim=-1), torch.ones(2, 5), atol=1e-6))

    def test_attention_respects_mask(self):
        q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        k = q.clone()
        v = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
        mask = torch.tensor([[True, False], [True, True]])
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        self.assertTrue(torch.allclose(attn[0, 0], torch.tensor([1.0, 0.0]), atol=1e-6))

    def test_transformer_block_shape(self):
        block = TinyTransformerBlock(d_model=8, d_hidden=16)
        x = torch.randn(3, 6, 8)
        y = block(x, mask=make_causal_mask(6))
        self.assertEqual(tuple(y.shape), (3, 6, 8))

    def test_masked_cross_entropy(self):
        logits = torch.tensor([[[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]]])
        targets = torch.tensor([[0, 1, -100]])
        got = masked_cross_entropy(logits, targets, ignore_index=-100)
        expected = torch.nn.functional.cross_entropy(logits[:, :2, :].reshape(-1, 2), targets[:, :2].reshape(-1))
        self.assertTrue(torch.allclose(got, expected))


if __name__ == '__main__':
    unittest.main()
