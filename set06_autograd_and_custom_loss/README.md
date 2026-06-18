# Exercise Set 6 — Autograd, Custom Losses, and Custom `Function`

## Goal
More advanced PyTorch mechanics:
- manual gradient reasoning,
- numerically stable loss implementation,
- writing a custom `torch.autograd.Function`,
- gradient checking.

## Tasks
1. Implement `stable_bce_with_logits` without calling `torch.nn.functional.binary_cross_entropy_with_logits`.
2. Implement a custom autograd function `SwishFunction`.
3. Implement `grad_norm` for a list of parameters.
4. Implement `finite_difference_check` for a scalar-valued function.

## Hints
- For numerically stable BCE with logits, use a formulation based on `max(x, 0)` and `log1p(exp(-abs(x)))`.
- In custom autograd functions, save tensors in `ctx.save_for_backward(...)`.
- Backward must return one gradient per forward argument.
- For finite differences, use central differences.

## Suggested References
- PyTorch autograd basics: https://pytorch.org/docs/stable/autograd.html
- Custom autograd functions: https://pytorch.org/docs/stable/notes/extending.html
- BCEWithLogitsLoss: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

  ## Опис виконання українською

У цьому завданні реалізовано розширені механізми PyTorch: чисельно стійку функцію втрат, власну autograd-функцію, обчислення норми градієнтів і перевірку градієнтів методом скінченних різниць.

У функції `stable_bce_with_logits` вручну реалізовано чисельно стійку формулу binary cross entropy with logits без використання `torch.nn.functional.binary_cross_entropy_with_logits`.

У класі `SwishFunction` реалізовано власну autograd-функцію для активації Swish за формулою `x * sigmoid(x)`. У методі `forward` обчислюється значення функції та зберігаються потрібні тензори, а в `backward` реалізовано аналітичний градієнт.

У функції `grad_norm` обчислено глобальну L2-норму градієнтів для параметрів, які мають ненульове поле `grad`.

У функції `finite_difference_check` реалізовано наближене обчислення градієнта скалярної функції методом центральних різниць.

Коментарі в коді додано українською мовою біля виконаних частин TODO.## Опис виконання українською

У цьому завданні реалізовано розширені механізми PyTorch: чисельно стійку функцію втрат, власну autograd-функцію, обчислення норми градієнтів і перевірку градієнтів методом скінченних різниць.

У функції `stable_bce_with_logits` вручну реалізовано чисельно стійку формулу binary cross entropy with logits без використання `torch.nn.functional.binary_cross_entropy_with_logits`.

У класі `SwishFunction` реалізовано власну autograd-функцію для активації Swish за формулою `x * sigmoid(x)`. У методі `forward` обчислюється значення функції та зберігаються потрібні тензори, а в `backward` реалізовано аналітичний градієнт.

У функції `grad_norm` обчислено глобальну L2-норму градієнтів для параметрів, які мають ненульове поле `grad`.

У функції `finite_difference_check` реалізовано наближене обчислення градієнта скалярної функції методом центральних різниць.

Коментарі в коді додано українською мовою біля виконаних частин TODO.
