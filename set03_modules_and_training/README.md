# Exercise Set 3 — Datasets, DataLoader, batching, and classification

# Goal
This set covers:
- custom `Dataset`
- `__len__`
- `__getitem__`
- `DataLoader`
- minibatch training for classification

## Tasks

1. Implement `ToyClassificationDataset`.
2. Implement `make_loader()`.
3. Implement `SimpleClassifier`.
4. Implement `accuracy_from_logits()`.
5. Implement one training epoch in `train_one_epoch()`.

## Hints

- Return `(features, label)` from the dataset.
- Labels for `CrossEntropyLoss` should be integer class indices (`torch.long`).
- `CrossEntropyLoss` expects raw logits, not softmax probabilities.
- Accuracy can be computed using `argmax(dim=1)`.

## References

- Dataset and DataLoader: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
- `torch.utils.data.Dataset`: https://pytorch.org/docs/stable/data.html
- `CrossEntropyLoss`: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

## Опис виконання українською

У цьому завданні реалізовано роботу з `Dataset`, `DataLoader`, мінібатчами та навчанням простої моделі класифікації в PyTorch.

У класі `ToyClassificationDataset` створено синтетичний датасет для бінарної класифікації. Ознаки збережено у форматі `float32`, а мітки класів — у форматі `long`, що потрібно для `CrossEntropyLoss`.

У функції `make_loader` створено `DataLoader`, який формує мінібатчі та за потреби перемішує дані.

У класі `SimpleClassifier` реалізовано просту нейронну мережу з двома лінійними шарами та функцією активації `ReLU`.

У функції `accuracy_from_logits` обчислено точність класифікації через `argmax(dim=1)`, оскільки модель повертає логіти класів.

У функції `train_one_epoch` реалізовано одну епоху навчання: обнулення градієнтів, прямий прохід, обчислення loss, зворотне поширення помилки та оновлення параметрів моделі.

Коментарі в коді додано українською мовою для пояснення основних етапів виконання.
