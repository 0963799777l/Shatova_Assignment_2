import torch


class MLPBlock(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout_p: float = 0.0):
        super().__init__()
        """
        Build a small MLP block: Linear -> ReLU -> Dropout -> Linear.
        """
        # TODO: define layers.
        # Створюємо послідовність шарів MLP-блоку.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(hidden_features, out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement the forward pass.
        # Передаємо вхідний тензор через усі шари моделі.
        return self.net(x)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count parameters with requires_grad=True.
    """
    # TODO: return an integer.
    # Рахуємо тільки параметри, які беруть участь у навчанні.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def xavier_init_(module: torch.nn.Module) -> None:
    """
    Apply Xavier uniform initialization to all Linear layers and zero their biases.
    """
    # TODO: modify the module in-place.
    # Проходимо по всіх шарах модуля і застосовуємо ініціалізацію до Linear-шарів.
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

            # Занулюємо bias, якщо він існує.
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    """
    Set requires_grad for all parameters in a module.
    """
    # TODO: implement this utility.
    # Встановлюємо requires_grad для всіх параметрів модуля.
    for p in module.parameters():
        p.requires_grad_(value)


def train_step(model, optimizer, criterion, x, y) -> float:
    """
    Do one training step and return the scalar loss as a Python float.
    """
    # TODO: zero grads, forward, compute loss, backward, step.
    # Переводимо модель у режим навчання.
    model.train()

    # Обнуляємо попередні градієнти.
    optimizer.zero_grad()

    # Виконуємо прямий прохід.
    pred = model(x)

    # Обчислюємо функцію втрат.
    loss = criterion(pred, y)

    # Обчислюємо градієнти.
    loss.backward()

    # Оновлюємо параметри моделі.
    optimizer.step()

    # Повертаємо loss як звичайне число Python.
    return loss.item()


def test_forward_shape():
    model = MLPBlock(4, 8, 2, dropout_p=0.0)
    x = torch.randn(5, 4)
    y = model(x)
    assert tuple(y.shape) == (5, 2)

def test_count_trainable_parameters():
    model = MLPBlock(3, 5, 2, dropout_p=0.0)
    got = count_trainable_parameters(model)
    expected = (3 * 5 + 5) + (5 * 2 + 2)
    assert got == expected


def test_xavier_init_biases_zero():
    model = MLPBlock(4, 8, 2, dropout_p=0.0)
    xavier_init_(model)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            assert torch.allclose(m.bias, torch.zeros_like(m.bias))


def test_set_requires_grad():
    model = MLPBlock(4, 8, 2, dropout_p=0.0)
    set_requires_grad(model, False)
    assert all(not p.requires_grad for p in model.parameters())
    set_requires_grad(model, True)
    assert all(p.requires_grad for p in model.parameters())


def test_train_step_decreases_loss_eventually():
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    true_w = torch.tensor([[2.0], [-1.0], [0.5], [3.0]])
    y = x @ true_w + 0.1
    model = MLPBlock(4, 16, 1, dropout_p=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    losses = [train_step(model, optimizer, criterion, x, y) for _ in range(30)]
    assert losses[-1] < losses[0]


if __name__ == '__main__':
    test_forward_shape()
    test_count_trainable_parameters()
    test_xavier_init_biases_zero()
    test_set_requires_grad()
    test_train_step_decreases_loss_eventually()
