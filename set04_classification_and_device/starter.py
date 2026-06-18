"""
Exercise Set 4: Convolutions, CNN blocks, and evaluation mode
Run:
    python exercise_set_4.py
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SmallCNN(nn.Module):
    """
    Input: (N, 1, 8, 8)
    Output: logits for 2 classes

    build a small CNN with conv, relu, pool, flatten, linear
    """
    def __init__(self):
        super().__init__()
        # TODO: define feature extractor and classifier
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward pass
        raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    """
    count trainable parameters only.
    """
    # TODO: sum numel() for parameters with requires_grad=True
    raise NotImplementedError


def predict_classes(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    run inference in eval mode without tracking gradients.
    """
    # TODO: switch to eval, use torch.no_grad, return predicted class indices
    raise NotImplementedError


def compute_batch_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    # TODO: compute batch accuracy
    raise NotImplementedError


def make_synthetic_images(n: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Class 0: bright left half
    Class 1: bright right half
    """
    torch.manual_seed(0)
    x = torch.zeros(n, 1, 8, 8)
    y = torch.randint(0, 2, (n,))
    for i in range(n):
        if y[i] == 0:
            x[i, 0, :, :4] = 1.0
        else:
            x[i, 0, :, 4:] = 1.0
        x[i] += 0.05 * torch.randn(1, 8, 8)
    return x, y


def train_cnn_steps(steps: int = 40, lr: float = 0.1) -> tuple[nn.Module, list[float]]:
    """
    train on the synthetic image task and return model + loss history.
    """
    x, y = make_synthetic_images(128)
    model = SmallCNN()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # TODO: implement short full-batch training loop
    raise NotImplementedError


def _test_forward_shape():
    model = SmallCNN()
    x = torch.randn(4, 1, 8, 8)
    out = model(x)
    assert out.shape == (4, 2)


def _test_count_parameters():
    model = SmallCNN()
    n = count_parameters(model)
    assert isinstance(n, int)
    assert n > 0


def _test_predict_classes():
    model = SmallCNN()
    x = torch.randn(3, 1, 8, 8)
    pred = predict_classes(model, x)
    assert pred.shape == (3,)
    assert pred.dtype == torch.long


def _test_training():
    model, history = train_cnn_steps(steps=60, lr=0.2)
    assert history[-1] < history[0]
    x, y = make_synthetic_images(64)
    pred = predict_classes(model, x)
    acc = float((pred == y).float().mean().item())
    assert acc > 0.9


def run_tests():
    _test_forward_shape()
    _test_count_parameters()
    _test_predict_classes()
    _test_training()
    print("All tests passed for Exercise Set 4.")


if __name__ == "__main__":
    run_tests()
