"""
Exercise Set 1: Tensors, indexing, broadcasting, and autograd
Run:
    python exercise_set_1.py

Students: complete the TODO sections.
Студенти: заповніть частини з TODO.
"""

import torch


def make_tensor() -> torch.Tensor:
    """
    Return a float32 tensor with values:
    [[1, 2, 3],
     [4, 5, 6]]

    EN: create the tensor exactly as specified.
    UA: створіть тензор точно у вказаному вигляді.
    """
    # TODO(EN): return the required tensor of dtype torch.float32
    # TODO(UA): поверніть потрібний тензор з типом torch.float32
    raise NotImplementedError


def row_means(x: torch.Tensor) -> torch.Tensor:
    """
    Return mean of each row.

    EN: input shape is (N, M), output shape must be (N,).
    UA: вхід має форму (N, M), вихід повинен мати форму (N,).
    """
    # TODO(EN): compute row-wise means
    # TODO(UA): обчисліть середні значення по рядках
    raise NotImplementedError


def normalize_columns(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each column:
        (x - column_mean) / (column_std + 1e-6)

    EN: use broadcasting, do not use Python loops.
    UA: використайте broadcasting, не використовуйте цикли Python.
    """
    # TODO(EN): normalize columns with broadcasting
    # TODO(UA): нормалізуйте стовпці за допомогою broadcasting
    raise NotImplementedError


def positive_elements(x: torch.Tensor) -> torch.Tensor:
    """
    Return a 1D tensor containing only positive elements of x.

    EN: use boolean masking.
    UA: використайте булеву маску.
    """
    # TODO(EN): filter positive values
    # TODO(UA): відфільтруйте додатні значення
    raise NotImplementedError


def squared_error_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Return sum of squared differences:
        sum((x - y)^2)

    EN: output must be a scalar tensor.
    UA: результат повинен бути скалярним тензором.
    """
    # TODO(EN): implement squared error loss
    # TODO(UA): реалізуйте квадратичну помилку
    raise NotImplementedError


def gradient_wrt_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of sum((x - y)^2) with respect to x.

    EN: use autograd, not a manual derivative formula.
    UA: використайте autograd, а не ручну формулу похідної.
    """
    # TODO(EN): clone x if needed, enable gradients, backprop, return gradient
    # TODO(UA): за потреби скопіюйте x, увімкніть градієнти, виконайте backprop, поверніть градієнт
    raise NotImplementedError


def _test_make_tensor():
    x = make_tensor()
    assert isinstance(x, torch.Tensor)
    assert x.dtype == torch.float32
    assert x.shape == (2, 3)
    assert torch.allclose(x, torch.tensor([[1., 2., 3.], [4., 5., 6.]]))


def _test_row_means():
    x = torch.tensor([[1., 2., 3.], [4., 5., 7.]])
    out = row_means(x)
    expected = torch.tensor([2.0, 16.0 / 3.0])
    assert out.shape == (2,)
    assert torch.allclose(out, expected)


def _test_normalize_columns():
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    out = normalize_columns(x)
    col_means = out.mean(dim=0)
    assert torch.allclose(col_means, torch.zeros_like(col_means), atol=1e-5)
    # std(unbiased=False) is more stable for small teaching examples
    col_std = out.std(dim=0, unbiased=False)
    assert torch.allclose(col_std, torch.ones_like(col_std), atol=1e-4)


def _test_positive_elements():
    x = torch.tensor([[-2., 0., 1.5], [3.0, -4.0, 2.0]])
    out = positive_elements(x)
    assert out.ndim == 1
    assert torch.allclose(out, torch.tensor([1.5, 3.0, 2.0]))


def _test_squared_error_loss():
    x = torch.tensor([1., 2., 3.])
    y = torch.tensor([1., 0., 5.])
    out = squared_error_loss(x, y)
    assert out.ndim == 0
    assert torch.allclose(out, torch.tensor(8.0))


def _test_gradient_wrt_x():
    x = torch.tensor([1., 2., 3.])
    y = torch.tensor([0., 0., 1.])
    grad = gradient_wrt_x(x, y)
    expected = 2 * (x - y)
    assert torch.allclose(grad, expected)


def run_tests():
    _test_make_tensor()
    _test_row_means()
    _test_normalize_columns()
    _test_positive_elements()
    _test_squared_error_loss()
    _test_gradient_wrt_x()
    print("All tests passed for Exercise Set 1.")


if __name__ == "__main__":
    run_tests()
