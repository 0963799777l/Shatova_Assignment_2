"""
Exercise Set 5: Advanced PyTorch: custom autograd, hooks, and checkpointing
Run:
    python exercise_set_5.py
"""

import os
import tempfile

import torch
import torch.nn as nn


class SquarePlusOne(torch.autograd.Function):
    """
    Forward:
        y = x^2 + 1

    implement forward and backward.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # TODO: save what is needed for backward and return x^2 + 1
        # Зберігаємо x, бо він потрібен для обчислення градієнта у backward.
        ctx.save_for_backward(x)

        # Повертаємо результат функції y = x^2 + 1.
        return x ** 2 + 1

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: return gradient with respect to x
        # Отримуємо збережений тензор x із forward.
        (x,) = ctx.saved_tensors

        # Для y = x^2 + 1 похідна дорівнює 2x.
        grad_x = grad_output * 2 * x

        return grad_x


def square_plus_one(x: torch.Tensor) -> torch.Tensor:
    # TODO: apply the custom autograd function
    # Застосовуємо власну autograd-функцію до тензора x.
    return SquarePlusOne.apply(x)


def freeze_module(module: nn.Module) -> None:
    """
    disable gradients for all parameters in the module.
    """
    # TODO: set requires_grad_(False) for all parameters
    # Вимикаємо обчислення градієнтів для всіх параметрів модуля.
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def save_checkpoint(model: nn.Module, path: str) -> None:
    """
    save model state_dict to path.
    """
    # TODO: use torch.save(...)
    # Зберігаємо лише state_dict моделі.
    torch.save(model.state_dict(), path)


def load_checkpoint(model: nn.Module, path: str) -> None:
    """
    load model state_dict from path.
    """
    # TODO: use torch.load(...) and load_state_dict(...)
    # Завантажуємо state_dict із файлу.
    state_dict = torch.load(path)

    # Передаємо завантажені параметри у модель.
    model.load_state_dict(state_dict)


def capture_activation_mean(module: nn.Module, x: torch.Tensor) -> float:
    """
    register a forward hook, run module(x), capture mean of output activation, remove hook, return mean.
    """
    # TODO: use register_forward_hook
    # Створюємо словник для збереження середнього значення активації.
    activation = {}

    # Hook викликається під час forward і зчитує вихід модуля.
    def hook_fn(module, input, output):
        activation["mean"] = output.mean().item()

    # Реєструємо forward hook.
    handle = module.register_forward_hook(hook_fn)

    try:
        # Виконуємо forward без обчислення градієнтів.
        with torch.no_grad():
            module(x)
    finally:
        # Обов'язково видаляємо hook після використання.
        handle.remove()

    return activation["mean"]


def _test_custom_autograd_forward():
    x = torch.tensor([2.0, -3.0])
    y = square_plus_one(x)
    assert torch.allclose(y, torch.tensor([5.0, 10.0]))


def _test_custom_autograd_backward():
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    y = square_plus_one(x).sum()
    y.backward()
    assert torch.allclose(x.grad, torch.tensor([4.0, -6.0]))


def _test_freeze_module():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
    freeze_module(model)
    assert all(not p.requires_grad for p in model.parameters())


def _test_checkpoint():
    model1 = nn.Linear(3, 2)
    with torch.no_grad():
        model1.weight.fill_(1.23)
        model1.bias.fill_(-0.5)

    model2 = nn.Linear(3, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        save_checkpoint(model1, path)
        load_checkpoint(model2, path)

    assert torch.allclose(model1.weight, model2.weight)
    assert torch.allclose(model1.bias, model2.bias)


def _test_capture_activation_mean():
    module = nn.Linear(4, 2)
    x = torch.randn(5, 4)
    value = capture_activation_mean(module, x)
    with torch.no_grad():
        expected = module(x).mean().item()
    assert abs(value - expected) < 1e-6


def run_tests():
    _test_custom_autograd_forward()
    _test_custom_autograd_backward()
    _test_freeze_module()
    _test_checkpoint()
    _test_capture_activation_mean()
    print("All tests passed for Exercise Set 5.")


if __name__ == "__main__":
    run_tests()
