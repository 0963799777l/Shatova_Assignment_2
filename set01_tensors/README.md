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

## Опис виконання українською

У цьому завданні реалізовано базові операції з тензорами PyTorch.

У функції `make_tensor` створено тензор типу `torch.float32` із заданими значеннями та формою `(2, 3)`.

У функції `row_means` обчислено середнє значення для кожного рядка тензора за допомогою операції редукції `mean`.

У функції `normalize_columns` виконано нормалізацію кожного стовпця за формулою `(x - column_mean) / (column_std + 1e-6)`. Для цього використано broadcasting без циклів Python.

У функції `positive_elements` реалізовано вибір лише додатних елементів тензора за допомогою булевої маски.

У функції `squared_error_loss` обчислено суму квадратів різниць між двома тензорами.

У функції `gradient_wrt_x` використано механізм autograd для обчислення градієнта функції втрат за тензором `x`.

Коментарі в коді додано українською мовою для пояснення основних етапів виконання.

