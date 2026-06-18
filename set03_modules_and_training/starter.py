"""
Exercise Set 3: Datasets, DataLoader, batching, and classification
Run:
    python exercise_set_3.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ToyClassificationDataset(Dataset):
    """
    Synthetic binary classification:
    label = 1 if x0 + x1 > 0 else 0

    EN: store features as float32 and labels as int64 / long.
    UA: збережіть ознаки як float32, а мітки як int64 / long.
    """
    def __init__(self, n_samples: int = 100):
        torch.manual_seed(0)
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).long()

        # TODO(EN): save tensors to self.x and self.y
        # TODO(UA): збережіть тензори в self.x і self.y
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO(EN): return dataset length
        # TODO(UA): поверніть довжину датасету
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO(EN): return one sample and one label
        # TODO(UA): поверніть один приклад і одну мітку
        raise NotImplementedError


def make_loader(dataset: Dataset, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    # TODO(EN): create and return DataLoader
    # TODO(UA): створіть і поверніть DataLoader
    raise NotImplementedError


class SimpleClassifier(nn.Module):
    """
    EN: a tiny MLP: Linear(2 -> hidden) + ReLU + Linear(hidden -> 2)
    UA: маленький MLP: Linear(2 -> hidden) + ReLU + Linear(hidden -> 2)
    """
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        # TODO(EN): define the network
        # TODO(UA): визначте мережу
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(EN): return logits
        # TODO(UA): поверніть логіти
        raise NotImplementedError


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    EN: compute classification accuracy.
    UA: обчисліть точність класифікації.
    """
    # TODO(EN): use argmax over class dimension
    # TODO(UA): використайте argmax по виміру класів
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    EN: train for one epoch, return average loss over batches.
    UA: натренуйте одну епоху, поверніть середній loss по батчах.
    """
    # TODO(EN): iterate over loader and perform training
    # TODO(UA): пройдіться по loader і виконайте навчання
    raise NotImplementedError


def _test_dataset():
    ds = ToyClassificationDataset(20)
    assert len(ds) == 20
    x, y = ds[0]
    assert x.shape == (2,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long


def _test_loader():
    ds = ToyClassificationDataset(20)
    loader = make_loader(ds, batch_size=5, shuffle=False)
    xb, yb = next(iter(loader))
    assert xb.shape == (5, 2)
    assert yb.shape == (5,)


def _test_classifier_shape():
    model = SimpleClassifier(hidden_dim=10)
    x = torch.randn(4, 2)
    logits = model(x)
    assert logits.shape == (4, 2)


def _test_accuracy():
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.5], [0.8, 0.4]])
    y = torch.tensor([0, 1, 1])
    acc = accuracy_from_logits(logits, y)
    assert abs(acc - (2/3)) < 1e-6


def _test_training():
    ds = ToyClassificationDataset(200)
    loader = make_loader(ds, batch_size=32, shuffle=True)
    model = SimpleClassifier(hidden_dim=16)
    optimizer = optim.SGD(model.parameters(), lr=0.2)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _ in range(15):
        losses.append(train_one_epoch(model, loader, optimizer, criterion))

    assert losses[-1] < losses[0]

    xb, yb = next(iter(loader))
    with torch.no_grad():
        logits = model(xb)
    acc = accuracy_from_logits(logits, yb)
    assert acc > 0.7


def run_tests():
    _test_dataset()
    _test_loader()
    _test_classifier_shape()
    _test_accuracy()
    _test_training()
    print("All tests passed for Exercise Set 3.")


if __name__ == "__main__":
    run_tests()
