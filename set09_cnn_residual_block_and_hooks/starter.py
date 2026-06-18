import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        """
        Build a residual block with two 3x3 convolutions and ReLU.
        """
        # TODO: define main path and skip path.
        # Створюємо основний шлях із двома згортками 3x3.
        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # Якщо кількість каналів змінюється, використовуємо 1x1 згортку для skip-шляху.
        if in_channels != out_channels:
            self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = torch.nn.Identity()

        # Фінальна активація після додавання residual.
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement residual addition and output activation.
        # Додаємо результат основного шляху та skip-з'єднання.
        out = self.main(x) + self.skip(x)

        # Застосовуємо ReLU після residual-додавання.
        return self.relu(out)


def global_avg_pool2d(x: torch.Tensor) -> torch.Tensor:
    """
    Average over H and W. Input: (B, C, H, W), output: (B, C).
    """
    # TODO: implement spatial global average pooling.
    # Усереднюємо тензор по висоті та ширині.
    return x.mean(dim=(2, 3))


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    """
    # TODO: return a Python float in [0, 1].
    # Визначаємо передбачений клас як індекс найбільшого логіта.
    preds = logits.argmax(dim=1)

    # Повертаємо середню частку правильних відповідей.
    return (preds == labels).float().mean()


def capture_activations(model: torch.nn.Module, layer_name: str, x: torch.Tensor) -> torch.Tensor:
    """
    Run the model once and capture the activation of the named submodule using a forward hook.
    """
    # TODO: register hook, run forward, remove hook, return captured activation.
    # Знаходимо потрібний підмодуль за його назвою.
    modules = dict(model.named_modules())
    layer = modules[layer_name]

    # Створюємо словник для збереження активації.
    activations = {}

    # Hook зберігає вихід вказаного шару під час forward.
    def hook_fn(module, input, output):
        activations["value"] = output.detach()

    # Реєструємо forward hook.
    handle = layer.register_forward_hook(hook_fn)

    try:
        # Запускаємо модель один раз без обчислення градієнтів.
        with torch.no_grad():
            model(x)
    finally:
        # Видаляємо hook після використання.
        handle.remove()

    return activations["value"]


class TinyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ResidualBlock(3, 8)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, x):
        x = self.block(x)
        x = global_avg_pool2d(x)
        x = self.head(x)
        return x


def test_residual_block_shape_same_channels():
    block = ResidualBlock(4, 4)
    x = torch.randn(2, 4, 16, 16)
    y = block(x)
    assert tuple(y.shape) == (2, 4, 16, 16)


def test_residual_block_shape_channel_change():
    block = ResidualBlock(3, 8)
    x = torch.randn(2, 3, 16, 16)
    y = block(x)
    assert tuple(y.shape) == (2, 8, 16, 16)


def test_global_avg_pool2d():
    x = torch.tensor([[[[1.0, 3.0], [5.0, 7.0]]]])
    got = global_avg_pool2d(x)
    expected = torch.tensor([[4.0]])
    assert torch.allclose(got, expected)


def test_compute_accuracy():
    logits = torch.tensor([[1.0, 3.0], [5.0, 4.0], [0.0, 1.0]])
    labels = torch.tensor([1, 0, 1])
    got = compute_accuracy(logits, labels)
    assert torch.allclose(got, torch.tensor(1.0))


def test_capture_activations():
    model = TinyCNN()
    x = torch.randn(2, 3, 8, 8)
    act = capture_activations(model, 'block', x)
    assert tuple(act.shape) == (2, 8, 8, 8)


if __name__ == '__main__':
    test_residual_block_shape_same_channels()
    test_residual_block_shape_channel_change()
    test_global_avg_pool2d()
    test_compute_accuracy()
    test_capture_activations()
