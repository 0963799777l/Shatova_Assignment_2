# Exercise Set 2 — Linear models, modules, and training loops

# Goal
This set introduces:
- `nn.Module`
- parameters
- forward pass
- mean squared error
- SGD training loop

## Tasks

1. Implement `SimpleLinearModel` with one `nn.Linear` layer.
2. Implement `mse_loss_manual()`.
3. Implement `train_step()` with:
   - `optimizer.zero_grad()`
   - forward pass
   - loss computation
   - `loss.backward()`
   - `optimizer.step()`
4. Implement `fit_regression()` to train a model on synthetic data.

## Hints

- `nn.Linear(in_features, out_features)` already includes weight and bias.
- For MSE, compute `((pred - target) ** 2).mean()`.
- During training, return Python floats with `.item()` when needed.
- Repeating small train steps is enough on the synthetic dataset here.

## References

- `nn.Module`: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- `nn.Linear`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- SGD: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
- Loss functions: https://pytorch.org/docs/stable/nn.html#loss-functions

## Опис виконання українською

У цьому завданні реалізовано просту лінійну модель PyTorch та базовий цикл навчання.

У класі `SimpleLinearModel` створено модель з одним лінійним шаром `nn.Linear`. Цей шар автоматично містить ваги та зсув, які навчаються під час оптимізації.

У функції `mse_loss_manual` вручну реалізовано середньоквадратичну помилку за формулою `((pred - target) ** 2).mean()`.

У функції `train_step` реалізовано один крок навчання: обнулення градієнтів, прямий прохід моделі, обчислення loss, зворотне поширення помилки та оновлення параметрів optimizer.

У функції `fit_regression` створено синтетичні дані для залежності `y = 3x - 2`, модель, optimizer SGD та цикл навчання. Після навчання функція повертає натреновану модель і список значень loss.

Коментарі в коді додано українською мовою для пояснення основних етапів виконання.
