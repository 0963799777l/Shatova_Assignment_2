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

    store features as float32 and labels as int64 / long.
    """
    def __init__(self, n_samples: int = 100):
        torch.manual_seed(0)
        x = torch.randn(n_samples, 2)
        y = (x[:, 0] + x[:, 1] > 0).long()

        # TODO: save tensors to self.x and self.y
        # Зберігаємо ознаки та мітки у змінних об'єкта датасету.
        self.x = x.float()
        self.y = y.long()

    def __len__(self) -> int:
        # TODO: return dataset length
        # Повертаємо кількість прикладів у датасеті.
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: return one sample and one label
        # Повертаємо один елемент датасету за індексом.
        return self.x[idx], self.y[idx]


def make_loader(dataset: Dataset, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    # TODO: create and return DataLoader
    # Створюємо DataLoader для автоматичного формування батчів.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SimpleClassifier(nn.Module):
    """
    a tiny MLP: Linear(2 -> hidden) + ReLU + Linear(hidden -> 2)
    """
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        # TODO: define the network
        # Створюємо просту нейронну мережу з двома лінійними шарами.
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return logits
        # Передаємо вхідні дані через мережу та отримуємо логіти.
        return self.net(x)


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
   compute classification accuracy.
    """
    # TODO: use argmax over class dimension
    # Визначаємо передбачений клас як індекс найбільшого логіта.
    preds = logits.argmax(dim=1)

    # Обчислюємо частку правильних відповідей.
    return (preds == y).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    train for one epoch, return average loss over batches.
    """
    # TODO: iterate over loader and perform training
    # Переводимо модель у режим навчання.
    model.train()

    total_loss = 0.0
    num_batches = 0

    # Проходимо по всіх батчах даних.
    for xb, yb in loader:
        # Обнуляємо градієнти перед новим кроком навчання.
        optimizer.zero_grad()

        # Виконуємо прямий прохід моделі.
        logits = model(xb)

        # Обчислюємо функцію втрат.
        loss = criterion(logits, yb)

        # Обчислюємо градієнти.
        loss.backward()

        # Оновлюємо параметри моделі.
        optimizer.step()

        # Накопичуємо loss для обчислення середнього значення.
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


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
