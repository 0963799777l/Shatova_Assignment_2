"""
Exercise Set 2: Linear models, modules, and training loops
Run:
    python exercise_set_2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SimpleLinearModel(nn.Module):
    """
     A model with a single linear layer.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # TODO: create self.linear as nn.Linear(...)
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: define forward pass
        raise NotImplementedError


def mse_loss_manual(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    implement mean squared error manually.
    """
    # TODO: return ((pred - target) ** 2).mean()
    raise NotImplementedError


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """
    perform one SGD step and return the scalar loss as float.
    """
    # TODO: zero gradients, forward, compute loss, backward, optimizer step
    raise NotImplementedError


def fit_regression(steps: int = 200, lr: float = 0.05) -> tuple[nn.Module, list[float]]:
    """
    Fit y = 3x - 2 on synthetic data.

    return trained model and loss history.
    """
    torch.manual_seed(0)
    x = torch.linspace(-2, 2, 100).unsqueeze(1)
    y = 3 * x - 2

    # TODO: create model and optimizer, run training loop
    raise NotImplementedError


def _test_model_structure():
    model = SimpleLinearModel(4, 2)
    assert isinstance(model, nn.Module)
    x = torch.randn(5, 4)
    y = model(x)
    assert y.shape == (5, 2)


def _test_mse_loss_manual():
    pred = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([[1.0], [1.0], [5.0]])
    expected = ((pred - target) ** 2).mean()
    out = mse_loss_manual(pred, target)
    assert torch.allclose(out, expected)


def _test_train_step_decreases_loss():
    torch.manual_seed(0)
    model = SimpleLinearModel(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    x = torch.linspace(-1, 1, 50).unsqueeze(1)
    y = 4 * x + 1

    with torch.no_grad():
        initial_loss = mse_loss_manual(model(x), y).item()

    for _ in range(50):
        train_step(model, optimizer, x, y)

    with torch.no_grad():
        final_loss = mse_loss_manual(model(x), y).item()

    assert final_loss < initial_loss * 0.2


def _test_fit_regression():
    model, history = fit_regression(steps=150, lr=0.1)
    assert len(history) == 150
    assert history[-1] < history[0]
    with torch.no_grad():
        pred = model(torch.tensor([[2.0]]))
    assert abs(pred.item() - 4.0) < 0.3


def run_tests():
    _test_model_structure()
    _test_mse_loss_manual()
    _test_train_step_decreases_loss()
    _test_fit_regression()
    print("All tests passed for Exercise Set 2.")


if __name__ == "__main__":
    run_tests()
