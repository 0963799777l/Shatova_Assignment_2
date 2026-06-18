# Exercise Set 7 — Custom Modules, Initialization, and Training Utilities

## Goal
Building reusable PyTorch components:
- a custom MLP block,
- parameter counting,
- Xavier initialization,
- freezing and unfreezing parameters,
- one training step.

## Tasks
1. Implement `MLPBlock` with `Linear -> ReLU -> Dropout -> Linear`.
2. Implement `count_trainable_parameters`.
3. Implement `xavier_init_` for linear layers.
4. Implement `set_requires_grad`.
5. Implement `train_step`.

## Hints
- Use `torch.nn.init.xavier_uniform_`.
- `optimizer.zero_grad()` should usually happen before backward.
- In `train_step`, return a Python float loss value.

## Suggested References
- `torch.nn.Module`: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- Initialization: https://pytorch.org/docs/stable/nn.init.html
- Optimizers: https://pytorch.org/docs/stable/optim.html

## Опис виконання українською

У цьому завданні реалізовано користувацький MLP-блок, ініціалізацію параметрів, керування навчанням параметрів і один крок тренування моделі.

У класі `MLPBlock` створено невелику нейронну мережу за структурою `Linear -> ReLU -> Dropout -> Linear`.

У функції `count_trainable_parameters` підраховано кількість параметрів моделі, для яких `requires_grad=True`.

У функції `xavier_init_` застосовано Xavier uniform ініціалізацію до всіх `Linear`-шарів, а їхні bias занулено.

У функції `set_requires_grad` реалізовано вмикання або вимикання обчислення градієнтів для всіх параметрів модуля.

У функції `train_step` реалізовано один крок навчання: обнулення градієнтів, прямий прохід, обчислення loss, backward і оновлення параметрів optimizer.

Коментарі в коді додано українською мовою біля виконаних частин TODO.
