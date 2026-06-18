# Exercise Set 9 — CNN Blocks, Residual Connections, and Forward Hooks

## Goal
Implement common CNN utilities and inspect activations:
- residual block,
- global average pooling,
- feature extraction with hooks,
- accuracy computation.

## Tasks
1. Implement `ResidualBlock` with two convolutions and a skip connection.
2. Implement `global_avg_pool2d`.
3. Implement `compute_accuracy`.
4. Implement `capture_activations` using a forward hook.

## Hints
- Use `padding=1` for 3x3 convolutions to preserve spatial size.
- If input and output channels differ, use a 1x1 convolution in the skip path.
- Forward hooks can append activations into a dict or list.

## Suggested References
- Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- Hooks: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
