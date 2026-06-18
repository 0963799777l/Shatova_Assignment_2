# Exercise Set 1 — Tensors, indexing, broadcasting, and autograd

# Goal
This set covers the PyTorch basics you should know after the NumPy assignment:
- creating tensors
- dtype and shape handling
- indexing and masking
- broadcasting
- reductions
- basic autograd

## Tasks

1. Implement `make_tensor()` to create a `float32` tensor of shape `(2, 3)`.
2. Implement `row_means()` using tensor reductions.
3. Implement `normalize_columns()` using broadcasting.
4. Implement `positive_elements()` using boolean masking.
5. Implement `squared_error_loss()` and `gradient_wrt_x()` using autograd.

## Hints

- Use `torch.tensor`, `torch.float32`, `mean`, boolean masks, and `sum`.
- For column normalization, keep dimensions where convenient.
- For gradients, make sure the input tensor has `requires_grad=True`.
- For scalar losses, autograd works most naturally when the result is a single value.

## References

- PyTorch tensors: https://pytorch.org/docs/stable/tensors.html
- Tensor operations: https://pytorch.org/docs/stable/torch.html
- Broadcasting semantics: https://pytorch.org/docs/stable/notes/broadcasting.html
- Autograd basics: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
