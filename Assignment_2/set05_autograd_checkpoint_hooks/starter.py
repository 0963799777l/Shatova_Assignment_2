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

    EN: implement forward and backward.
    UA: реалізуйте forward і backward.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        # TODO(EN): save what is needed for backward and return x^2 + 1
        # TODO(UA): збережіть потрібне для backward і поверніть x^2 + 1
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO(EN): return gradient with respect to x
        # TODO(UA): поверніть градієнт по x
        raise NotImplementedError


def square_plus_one(x: torch.Tensor) -> torch.Tensor:
    # TODO(EN): apply the custom autograd function
    # TODO(UA): застосуйте власну autograd function
    raise NotImplementedError


def freeze_module(module: nn.Module) -> None:
    """
    EN: disable gradients for all parameters in the module.
    UA: вимкніть градієнти для всіх параметрів модуля.
    """
    # TODO(EN): set requires_grad_(False) for all parameters
    # TODO(UA): встановіть requires_grad_(False) для всіх параметрів
    raise NotImplementedError


def save_checkpoint(model: nn.Module, path: str) -> None:
    """
    EN: save model state_dict to path.
    UA: збережіть state_dict моделі у path.
    """
    # TODO(EN): use torch.save(...)
    # TODO(UA): використайте torch.save(...)
    raise NotImplementedError


def load_checkpoint(model: nn.Module, path: str) -> None:
    """
    EN: load model state_dict from path.
    UA: завантажте state_dict моделі з path.
    """
    # TODO(EN): use torch.load(...) and load_state_dict(...)
    # TODO(UA): використайте torch.load(...) та load_state_dict(...)
    raise NotImplementedError


def capture_activation_mean(module: nn.Module, x: torch.Tensor) -> float:
    """
    EN: register a forward hook, run module(x), capture mean of output activation, remove hook, return mean.
    UA: зареєструйте forward hook, виконайте module(x), зчитайте середнє значення виходу, зніміть hook, поверніть mean.
    """
    # TODO(EN): use register_forward_hook
    # TODO(UA): використайте register_forward_hook
    raise NotImplementedError


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
