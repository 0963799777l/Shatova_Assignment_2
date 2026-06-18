# Exercise Set 8 — Datasets, DataLoaders, Padding, and Masks

## Goal
Implement a small variable-length sequence pipeline:
- custom dataset,
- custom collate function,
- padding,
- boolean masks,
- masked mean pooling.

## Tasks
1. Implement `ToySequenceDataset`.
2. Implement `pad_collate`.
3. Implement `masked_mean`.
4. Implement `sequence_regression_step`.

## Hints
- The dataset can return `(sequence_tensor, target_tensor)` pairs.
- In the collate function, output tensors with shape `(batch, max_len, feature_dim)` and a mask of shape `(batch, max_len)`.
- The mask should be `True` for valid tokens.

## Suggested References
- Dataset and DataLoader: https://pytorch.org/docs/stable/data.html
- Padding utilities: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
