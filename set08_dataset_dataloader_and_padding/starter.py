import unittest
import torch
from torch.utils.data import Dataset, DataLoader


class ToySequenceDataset(Dataset):
    def __init__(self, sequences, targets):
        """
        EN: Store variable-length sequences and corresponding targets.
        UA: Збережіть послідовності змінної довжини та відповідні цілі.
        """
        # TODO(EN): store the data.
        # TODO(UA): збережіть дані.
        raise NotImplementedError

    def __len__(self):
        # TODO(EN): return dataset length.
        # TODO(UA): поверніть довжину датасету.
        raise NotImplementedError

    def __getitem__(self, idx):
        # TODO(EN): return (sequence, target).
        # TODO(UA): поверніть (sequence, target).
        raise NotImplementedError


def pad_collate(batch):
    """
    EN: Collate a batch of variable-length sequences.
    Return:
        padded_x: (B, T, D)
        mask: (B, T) boolean, True where data is valid
        y: (B, ...)

    UA: Зберіть batch зі змінною довжиною послідовностей.
    Поверніть:
        padded_x: (B, T, D)
        mask: (B, T) boolean, True там, де дані валідні
        y: (B, ...)
    """
    # TODO(EN): pad sequences to the max length in the batch.
    # TODO(UA): доповніть послідовності до максимальної довжини в batch.
    raise NotImplementedError


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    EN: Compute mean over time dimension using the mask.
    UA: Обчисліть середнє по часовому виміру з використанням mask.
    """
    # TODO(EN): output shape should be (B, D).
    # TODO(UA): форма виходу має бути (B, D).
    raise NotImplementedError


def sequence_regression_step(model, optimizer, batch) -> float:
    """
    EN: One training step for sequence regression with masked mean pooling.
    UA: Один крок навчання для регресії по послідовностях із masked mean pooling.
    """
    # TODO(EN): compute pooled features, predict, loss, backward, step.
    # TODO(UA): обчисліть pooled-ознаки, прогноз, loss, backward, step.
    raise NotImplementedError


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
