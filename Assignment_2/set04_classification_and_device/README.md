# Exercise Set 4 — Convolutions, CNN blocks, and evaluation mode

# Goal

This set introduces:
- 2D convolutions
- flattening feature maps
- small CNN architecture
- dropout
- train vs eval mode
- inference with `torch.no_grad()`

## Tasks

1. Implement `SmallCNN`.
2. Implement `count_parameters()`.
3. Implement `predict_classes()` in eval mode with `torch.no_grad()`.
4. Implement `compute_batch_accuracy()`.
5. Implement a short `train_cnn_steps()` routine on a synthetic image dataset.

## Hints

- Use `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten`, `nn.Linear`, `nn.Dropout`.
- Be careful with tensor shapes after convolution and pooling.
- Call `model.eval()` for inference and `model.train()` for training.
- Synthetic images are enough for testing architecture and loop correctness.

## References

- Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- MaxPool2d: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- no_grad: https://pytorch.org/docs/stable/generated/torch.no_grad.html
- Module training/eval mode: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
