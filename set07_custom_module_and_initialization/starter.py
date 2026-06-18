import torch


class MLPBlock(torch.nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout_p: float = 0.0):
        super().__init__()
        """
        EN: Build a small MLP block: Linear -> ReLU -> Dropout -> Linear.
        UA: Побудуйте невеликий MLP-блок: Linear -> ReLU -> Dropout -> Linear.
        """
        # TODO(EN): define layers.
        # TODO(UA): визначте шари.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(EN): implement the forward pass.
        # TODO(UA): реалізуйте прямий прохід.
        raise NotImplementedError


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    EN: Count parameters with requires_grad=True.
    UA: Порахуйте параметри, для яких requires_grad=True.
    """
    # TODO(EN): return an integer.
    # TODO(UA): поверніть ціле число.
    raise NotImplementedError


def xavier_init_(module: torch.nn.Module) -> None:
    """
    EN: Apply Xavier uniform initialization to all Linear layers and zero their biases.
    UA: Застосуйте Xavier uniform ініціалізацію до всіх Linear-шарів і занульте їх bias.
    """
    # TODO(EN): modify the module in-place.
    # TODO(UA): змініть модуль in-place.
    raise NotImplementedError


def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    """
    EN: Set requires_grad for all parameters in a module.
    UA: Встановіть requires_grad для всіх параметрів модуля.
    """
    # TODO(EN): implement this utility.
    # TODO(UA): реалізуйте цю утиліту.
    raise NotImplementedError


def train_step(model, optimizer, criterion, x, y) -> float:
    """
    EN: Do one training step and return the scalar loss as a Python float.
    UA: Виконайте один крок навчання і поверніть скалярне значення loss як Python float.
    """
    # TODO(EN): zero grads, forward, compute loss, backward, step.
    # TODO(UA): обнуліть градієнти, зробіть forward, обчисліть loss, backward, step.
    raise NotImplementedError


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
