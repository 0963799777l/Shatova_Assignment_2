import math
import unittest
import torch


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    """
    EN: Implement scaled dot-product attention.
    Shapes:
        q, k, v: (B, T, D)
        mask: (T, T) or (B, T, T), True where attention is allowed
    Return:
        output: (B, T, D)
        attn: (B, T, T)

    UA: Реалізуйте scaled dot-product attention.
    Форми:
        q, k, v: (B, T, D)
        mask: (T, T) або (B, T, T), True там, де увага дозволена
    Поверніть:
        output: (B, T, D)
        attn: (B, T, T)
    """
    # TODO(EN): compute scaled attention scores, apply mask, softmax, and weighted sum.
    # TODO(UA): обчисліть scaled scores, застосуйте mask, softmax і зважену суму.
    raise NotImplementedError


def make_causal_mask(seq_len: int) -> torch.Tensor:
    """
    EN: Return a boolean causal mask of shape (T, T), where position i can attend to <= i.
    UA: Поверніть булеву causal mask форми (T, T), де позиція i може дивитися лише на <= i.
    """
    # TODO(EN): create a lower-triangular boolean mask.
    # TODO(UA): створіть нижньотрикутну булеву маску.
    raise NotImplementedError


class TinyTransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        """
        EN: Build a tiny transformer-style block with:
            - q_proj, k_proj, v_proj
            - attention
            - residual + layernorm
            - feed-forward: Linear -> ReLU -> Linear
            - residual + layernorm
        UA: Побудуйте маленький transformer-подібний блок із:
            - q_proj, k_proj, v_proj
            - attention
            - residual + layernorm
            - feed-forward: Linear -> ReLU -> Linear
            - residual + layernorm
        """
        # TODO(EN): define submodules.
        # TODO(UA): визначте підмодулі.
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # TODO(EN): implement the transformer block forward pass.
        # TODO(UA): реалізуйте forward для transformer-блоку.
        raise NotImplementedError


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    EN: Compute token-level cross entropy while ignoring padding positions.
    logits: (B, T, C)
    targets: (B, T)

    UA: Обчисліть token-level cross entropy, ігноруючи padding-позиції.
    logits: (B, T, C)
    targets: (B, T)
    """
    # TODO(EN): flatten batch/time dimensions and call cross_entropy.
    # TODO(UA): розгорніть batch/time виміри і викличте cross_entropy.
    raise NotImplementedError


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
