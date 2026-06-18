# Exercise Set 6 — Autograd, Custom Losses, and Custom `Function`

## Goal
More advanced PyTorch mechanics:
- manual gradient reasoning,
- numerically stable loss implementation,
- writing a custom `torch.autograd.Function`,
- gradient checking.

## Tasks
1. Implement `stable_bce_with_logits` without calling `torch.nn.functional.binary_cross_entropy_with_logits`.
2. Implement a custom autograd function `SwishFunction`.
3. Implement `grad_norm` for a list of parameters.
4. Implement `finite_difference_check` for a scalar-valued function.

## Hints
- For numerically stable BCE with logits, use a formulation based on `max(x, 0)` and `log1p(exp(-abs(x)))`.
- In custom autograd functions, save tensors in `ctx.save_for_backward(...)`.
- Backward must return one gradient per forward argument.
- For finite differences, use central differences.

## Suggested References
- PyTorch autograd basics: https://pytorch.org/docs/stable/autograd.html
- Custom autograd functions: https://pytorch.org/docs/stable/notes/extending.html
- BCEWithLogitsLoss: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
