import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        """
        EN: Build a residual block with two 3x3 convolutions and ReLU.
        UA: Побудуйте residual-блок із двома 3x3 згортками та ReLU.
        """
        # TODO(EN): define main path and skip path.
        # TODO(UA): визначте основний шлях і skip-шлях.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(EN): implement residual addition and output activation.
        # TODO(UA): реалізуйте додавання residual і вихідну активацію.
        raise NotImplementedError


def global_avg_pool2d(x: torch.Tensor) -> torch.Tensor:
    """
    EN: Average over H and W. Input: (B, C, H, W), output: (B, C).
    UA: Усередніть по H і W. Вхід: (B, C, H, W), вихід: (B, C).
    """
    # TODO(EN): implement spatial global average pooling.
    # TODO(UA): реалізуйте глобальний average pooling по просторових вимірах.
    raise NotImplementedError


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    EN: Compute classification accuracy.
    UA: Обчисліть accuracy класифікації.
    """
    # TODO(EN): return a Python float in [0, 1].
    # TODO(UA): поверніть Python float у межах [0, 1].
    raise NotImplementedError


def capture_activations(model: torch.nn.Module, layer_name: str, x: torch.Tensor) -> torch.Tensor:
    """
    EN: Run the model once and capture the activation of the named submodule using a forward hook.
    UA: Один раз запустіть модель і збережіть активацію вказаного підмодуля через forward hook.
    """
    # TODO(EN): register hook, run forward, remove hook, return captured activation.
    # TODO(UA): зареєструйте hook, виконайте forward, зніміть hook, поверніть активацію.
    raise NotImplementedError


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
