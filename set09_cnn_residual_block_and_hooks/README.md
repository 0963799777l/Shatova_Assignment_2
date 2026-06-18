# Exercise Set 9 — CNN Blocks, Residual Connections, and Forward Hooks

## Goal
Implement common CNN utilities and inspect activations:
- residual block,
- global average pooling,
- feature extraction with hooks,
- accuracy computation.

## Tasks
1. Implement `ResidualBlock` with two convolutions and a skip connection.
2. Implement `global_avg_pool2d`.
3. Implement `compute_accuracy`.
4. Implement `capture_activations` using a forward hook.

## Hints
- Use `padding=1` for 3x3 convolutions to preserve spatial size.
- If input and output channels differ, use a 1x1 convolution in the skip path.
- Forward hooks can append activations into a dict or list.

## Suggested References
- Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- Hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html

## Опис виконання українською

У цьому завданні реалізовано CNN-блок із residual-з'єднанням, глобальний average pooling, обчислення точності класифікації та збереження активацій шару за допомогою forward hook.

У класі `ResidualBlock` створено residual-блок із двома згортковими шарами `Conv2d`, функцією активації `ReLU` та skip-з'єднанням. Якщо кількість вхідних і вихідних каналів відрізняється, у skip-шляху використано згортку `1x1`.

У функції `global_avg_pool2d` реалізовано усереднення по просторових вимірах `H` і `W`, у результаті чого тензор форми `(B, C, H, W)` перетворюється на `(B, C)`.

У функції `compute_accuracy` обчислено точність класифікації через `argmax` по виміру класів.

У функції `capture_activations` використано forward hook для збереження активації вказаного підмодуля під час одного forward-проходу моделі.

Коментарі в коді додано українською мовою біля виконаних частин TODO.
