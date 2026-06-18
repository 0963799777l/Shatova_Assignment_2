# Exercise Set 4 — Convolutions, CNN blocks, and evaluation mode

# Goal

This set introduces:
- 2D convolutions
- flattening feature maps
- small CNN architecture
- dropout
- train vs eval mode
- inference with `torch.no_grad()`

## Tasks

1. Implement `SmallCNN`.
2. Implement `count_parameters()`.
3. Implement `predict_classes()` in eval mode with `torch.no_grad()`.
4. Implement `compute_batch_accuracy()`.
5. Implement a short `train_cnn_steps()` routine on a synthetic image dataset.

## Hints

- Use `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten`, `nn.Linear`, `nn.Dropout`.
- Be careful with tensor shapes after convolution and pooling.
- Call `model.eval()` for inference and `model.train()` for training.
- Synthetic images are enough for testing architecture and loop correctness.

## References

- Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- MaxPool2d: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- no_grad: https://pytorch.org/docs/stable/generated/torch.no_grad.html
- Module training/eval mode: https://pytorch.org/docs/stable/generated/torch.nn.Module.html

## Опис виконання українською

У цьому завданні реалізовано невелику згорткову нейронну мережу для класифікації синтетичних зображень.

У класі `SmallCNN` створено CNN-архітектуру, яка складається зі згорткових шарів `Conv2d`, функцій активації `ReLU`, шарів `MaxPool2d`, операції `Flatten`, шару `Dropout` і фінального лінійного шару `Linear`.

У функції `count_parameters` підраховано кількість параметрів моделі, які беруть участь у навчанні.

У функції `predict_classes` реалізовано передбачення класів у режимі `eval` з використанням `torch.no_grad`, щоб не обчислювати градієнти під час inference.

У функції `compute_batch_accuracy` обчислено точність класифікації для одного батча.

У функції `train_cnn_steps` реалізовано короткий цикл навчання моделі на синтетичних зображеннях, де клас 0 має яскраву ліву половину, а клас 1 — яскраву праву половину.

Коментарі в коді додано українською мовою біля виконаних частин TODO.## Опис виконання українською

У цьому завданні реалізовано невелику згорткову нейронну мережу для класифікації синтетичних зображень.

У класі `SmallCNN` створено CNN-архітектуру, яка складається зі згорткових шарів `Conv2d`, функцій активації `ReLU`, шарів `MaxPool2d`, операції `Flatten`, шару `Dropout` і фінального лінійного шару `Linear`.

У функції `count_parameters` підраховано кількість параметрів моделі, які беруть участь у навчанні.

У функції `predict_classes` реалізовано передбачення класів у режимі `eval` з використанням `torch.no_grad`, щоб не обчислювати градієнти під час inference.

У функції `compute_batch_accuracy` обчислено точність класифікації для одного батча.

У функції `train_cnn_steps` реалізовано короткий цикл навчання моделі на синтетичних зображеннях, де клас 0 має яскраву ліву половину, а клас 1 — яскраву праву половину.

Коментарі в коді додано українською мовою біля виконаних частин TODO.
