"""
Exercise Set 1: Tensors, indexing, broadcasting, and autograd
Run:
    python exercise_set_1.py

Students: complete the TODO sections.
"""

import torch


def make_tensor() -> torch.Tensor:
    """
    Return a float32 tensor with values:
    [[1, 2, 3],
     [4, 5, 6]]

    EN: create the tensor exactly as specified.
    """
    # Створюємо тензор PyTorch із потрібними значеннями та типом float32.
    return torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)


def row_means(x: torch.Tensor) -> torch.Tensor:
    """
    Return mean of each row.

    EN: input shape is (N, M), output shape must be (N,).
    """
    # Обчислюємо середнє значення для кожного рядка.
    # dim=1 означає, що усереднення виконується по стовпцях у межах кожного рядка.
    return x.mean(dim=1)


def normalize_columns(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize each column:
        (x - column_mean) / (column_std + 1e-6)

    EN: use broadcasting, do not use Python loops.
    """
    # Обчислюємо середнє значення для кожного стовпця.
    # keepdim=True зберігає форму (1, M), щоб broadcasting працював коректно.
    column_mean = x.mean(dim=0, keepdim=True)

    # Обчислюємо стандартне відхилення для кожного стовпця.
    # unbiased=False використовується відповідно до тестів.
    column_std = x.std(dim=0, unbiased=False, keepdim=True)

    # Нормалізуємо кожен стовпець за формулою:
    # (x - column_mean) / (column_std + 1e-6)
    return (x - column_mean) / (column_std + 1e-6)


def positive_elements(x: torch.Tensor) -> torch.Tensor:
    """
    Return a 1D tensor containing only positive elements of x.

    EN: use boolean masking.
    """
    # Створюємо булеву маску x > 0 і вибираємо лише додатні елементи.
    return x[x > 0]


def squared_error_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Return sum of squared differences:
        sum((x - y)^2)

    EN: output must be a scalar tensor.
    """
    # Обчислюємо суму квадратів різниць між x та y.
    return torch.sum((x - y) ** 2)


def gradient_wrt_x(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute gradient of sum((x - y)^2) with respect to x.

    EN: use autograd, not a manual derivative formula.
    """
    # Клонуємо x, щоб не змінювати початковий тензор.
    # detach() від'єднує тензор від попереднього графа обчислень.
    # requires_grad_(True) вмикає обчислення градієнтів.
    x_var = x.clone().detach().requires_grad_(True)

    # Обчислюємо скалярну функцію втрат.
    loss = squared_error_loss(x_var, y)

    # Виконуємо зворотне поширення для обчислення градієнта.
    loss.backward()

    # Повертаємо градієнт за x.
    return x_var.grad


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
