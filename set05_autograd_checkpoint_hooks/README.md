# Exercise Set 5 — Advanced PyTorch: custom autograd, hooks, and checkpointing

# Goal
This set covers more advanced topics:
- custom `autograd.Function`
- forward / backward logic
- parameter hooks
- saving and loading model state
- freezing parameters

## Tasks

1. Implement a custom autograd function `SquarePlusOne`.
2. Wrap it in `square_plus_one()`.
3. Implement `freeze_module()`.
4. Implement `save_checkpoint()` and `load_checkpoint()`.
5. Implement `capture_activation_mean()` using a forward hook.

## Hints

- In `backward`, return gradients for each forward input.
- For `y = x^2 + 1`, the derivative is `2x`.
- `state_dict()` and `load_state_dict()` are standard checkpoint tools.
- Forward hooks can observe module input/output without changing the model.

## References

- Extending autograd: https://pytorch.org/docs/stable/notes/extending.html
- Saving and loading models: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- Hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html

## Опис виконання українською

У цьому завданні реалізовано розширені можливості PyTorch: власну autograd-функцію, роботу з градієнтами, збереження і завантаження стану моделі, заморожування параметрів та використання forward hook.

У класі `SquarePlusOne` реалізовано власну функцію `y = x^2 + 1` з окремими методами `forward` і `backward`. У `forward` зберігається тензор `x`, а у `backward` обчислюється градієнт за формулою `2x`.

У функції `square_plus_one` застосовано власну autograd-функцію через метод `apply`.

У функції `freeze_module` вимкнено обчислення градієнтів для всіх параметрів модуля.

У функціях `save_checkpoint` і `load_checkpoint` реалізовано збереження та завантаження `state_dict` моделі.

У функції `capture_activation_mean` використано forward hook для отримання середнього значення вихідної активації модуля.

Коментарі в коді додано українською мовою біля виконаних частин TODO.
