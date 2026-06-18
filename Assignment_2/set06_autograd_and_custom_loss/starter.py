import torch


def stable_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute mean binary cross entropy with logits in a numerically stable way.

    EN: Implement a stable BCE-with-logits formula manually.
    UA: Реалізуйте чисельно стійку формулу BCE-with-logits вручну.
    """
    # TODO(EN): return a scalar tensor with the mean loss.
    # TODO(UA): поверніть скалярний тензор із середнім значенням втрат.
    raise NotImplementedError


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """
        EN: Implement Swish: x * sigmoid(x)
        UA: Реалізуйте Swish: x * sigmoid(x)
        """
        # TODO(EN): compute output and save what is needed for backward.
        # TODO(UA): обчисліть вихід і збережіть усе необхідне для backward.
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        EN: Return gradient with respect to x.
        UA: Поверніть градієнт відносно x.
        """
        # TODO(EN): implement analytical backward for Swish.
        # TODO(UA): реалізуйте аналітичний backward для Swish.
        raise NotImplementedError


def swish(x: torch.Tensor) -> torch.Tensor:
    return SwishFunction.apply(x)


def grad_norm(parameters) -> torch.Tensor:
    """
    EN: Compute the global L2 norm of gradients over all parameters that have gradients.
    UA: Обчисліть глобальну L2-норму градієнтів для всіх параметрів, які мають градієнти.
    """
    # TODO(EN): ignore parameters with grad is None.
    # TODO(UA): ігноруйте параметри, у яких grad is None.
    raise NotImplementedError


def finite_difference_check(f, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    EN: Approximate df/dx for a scalar-valued function f using central differences.
    UA: Наближено обчисліть df/dx для скалярної функції f методом центральних різниць.
    """
    # TODO(EN): return tensor of same shape as x with finite-difference gradient.
    # TODO(UA): поверніть тензор тієї ж форми, що й x, із градієнтом через скінченні різниці.
    raise NotImplementedError


def test_stable_bce_matches_pytorch():
    torch.manual_seed(0)
    logits = torch.randn(32)
    targets = torch.randint(0, 2, (32,), dtype=torch.float32)
    got = stable_bce_with_logits(logits, targets)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(got, expected, atol=1e-6)


def test_swish_matches_reference():
    x = torch.linspace(-3, 3, 17, requires_grad=True)
    got = swish(x)
    expected = x * torch.sigmoid(x)
    assert torch.allclose(got, expected, atol=1e-6)


def test_swish_backward_matches_autograd():
    torch.manual_seed(0)
    x1 = torch.randn(11, requires_grad=True)
    y1 = swish(x1).sum()
    y1.backward()
    g1 = x1.grad.clone()

    x2 = x1.detach().clone().requires_grad_(True)
    y2 = (x2 * torch.sigmoid(x2)).sum()
    y2.backward()
    g2 = x2.grad.clone()

    assert torch.allclose(g1, g2, atol=1e-6)


def test_grad_norm():
    p1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    p2 = torch.nn.Parameter(torch.tensor([3.0]))
    p1.grad = torch.tensor([3.0, 4.0])
    p2.grad = torch.tensor([12.0])
    got = grad_norm([p1, p2])
    expected = torch.tensor(13.0)
    assert torch.allclose(got, expected)


def test_finite_difference_check():
    def f(z):
        return (z ** 2).sum()

    x = torch.tensor([1.5, -2.0, 0.5], dtype=torch.float64)
    got = finite_difference_check(f, x, eps=1e-6)
    expected = 2 * x
    assert torch.allclose(got, expected, atol=1e-4, rtol=1e-4)


if __name__ == '__main__':
    test_stable_bce_matches_pytorch()
    test_swish_matches_reference()
    test_swish_backward_matches_autograd()
    test_grad_norm()
    test_finite_difference_check()
