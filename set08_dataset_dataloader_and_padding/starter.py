import unittest
import torch
from torch.utils.data import Dataset, DataLoader


class ToySequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        """
        Store variable-length sequences and corresponding targets.
        """
        # TODO: store the data.
        # Зберігаємо послідовності змінної довжини та відповідні цільові значення.
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        # TODO: return dataset length.
        # Повертаємо кількість послідовностей у датасеті.
        return len(self.sequences)

    def __getitem__(self, idx):
        # TODO: return (sequence, target).
        # Повертаємо одну послідовність і відповідне їй цільове значення.
        return self.sequences[idx], self.targets[idx]


def pad_collate(batch):
    """
    Collate a batch of variable-length sequences.
    Return:
        padded_x: (B, T, D)
        mask: (B, T) boolean, True where data is valid
        y: (B, ...)
    """
    # TODO: pad sequences to the max length in the batch.
    # Розділяємо batch на послідовності та цільові значення.
    sequences, targets = zip(*batch)

    # Визначаємо розмір batch, максимальну довжину послідовності та кількість ознак.
    batch_size = len(sequences)
    max_len = max(seq.shape[0] for seq in sequences)
    feature_dim = sequences[0].shape[1]

    # Створюємо тензор із нулями для доповнених послідовностей.
    padded_x = torch.zeros(batch_size, max_len, feature_dim, dtype=sequences[0].dtype)

    # Створюємо булеву маску, де True відповідає реальним елементам послідовності.
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # Заповнюємо padded_x реальними значеннями послідовностей.
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded_x[i, :length] = seq
        mask[i, :length] = True

    # Об'єднуємо цільові значення в один тензор.
    y = torch.stack(targets, dim=0)

    return padded_x, mask, y


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean over time dimension using the mask.
    """
    # TODO: output shape should be (B, D).
    # Додаємо останній вимір до mask, щоб її можна було застосувати до x.
    mask_expanded = mask.unsqueeze(-1)

    # Обнуляємо padded-елементи, які не є валідними.
    masked_x = x * mask_expanded

    # Сумуємо тільки валідні елементи по часовому виміру.
    summed = masked_x.sum(dim=1)

    # Рахуємо кількість валідних елементів у кожній послідовності.
    counts = mask_expanded.sum(dim=1).clamp(min=1)

    # Ділимо суму на кількість валідних елементів.
    return summed / counts


def sequence_regression_step(model, optimizer, batch) -> float:
    """
    One training step for sequence regression with masked mean pooling.
    """
    # TODO: compute pooled features, predict, loss, backward, step.
    # Розпаковуємо batch.
    x, mask, y = batch

    # Переводимо модель у режим навчання.
    model.train()

    # Обнуляємо попередні градієнти.
    optimizer.zero_grad()

    # Отримуємо pooled-ознаки через masked mean.
    pooled = masked_mean(x, mask)

    # Виконуємо прогноз моделі.
    pred = model(pooled)

    # Обчислюємо MSE loss для регресії.
    loss = torch.nn.functional.mse_loss(pred, y)

    # Обчислюємо градієнти.
    loss.backward()

    # Оновлюємо параметри моделі.
    optimizer.step()

    # Повертаємо loss як звичайне число Python.
    return loss.item()


class TestDatasetAndPadding(unittest.TestCase):
    def make_dataset(self):
        seqs = [
            torch.tensor([[1.0], [2.0], [3.0]]),
            torch.tensor([[10.0], [20.0]]),
            torch.tensor([[5.0]]),
        ]
        targets = [torch.tensor([2.0]), torch.tensor([15.0]), torch.tensor([5.0])]
        return ToySequenceDataset(seqs, targets)

    def test_dataset_basic(self):
        ds = self.make_dataset()
        self.assertEqual(len(ds), 3)
        x, y = ds[1]
        self.assertEqual(tuple(x.shape), (2, 1))
        self.assertEqual(tuple(y.shape), (1,))

    def test_pad_collate_shapes(self):
        ds = self.make_dataset()
        batch = [ds[0], ds[1], ds[2]]
        x, mask, y = pad_collate(batch)
        self.assertEqual(tuple(x.shape), (3, 3, 1))
        self.assertEqual(tuple(mask.shape), (3, 3))
        self.assertEqual(tuple(y.shape), (3, 1))

    def test_masked_mean(self):
        x = torch.tensor([
            [[1.0], [2.0], [0.0]],
            [[10.0], [20.0], [30.0]],
        ])
        mask = torch.tensor([
            [True, True, False],
            [True, True, True],
        ])
        got = masked_mean(x, mask)
        expected = torch.tensor([[1.5], [20.0]])
        self.assertTrue(torch.allclose(got, expected))

    def test_sequence_regression_step(self):
        torch.manual_seed(0)
        ds = self.make_dataset()
        loader = DataLoader(ds, batch_size=3, collate_fn=pad_collate)
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        losses = []
        for _ in range(40):
            for batch in loader:
                losses.append(sequence_regression_step(model, opt, batch))
        self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
    unittest.main()
