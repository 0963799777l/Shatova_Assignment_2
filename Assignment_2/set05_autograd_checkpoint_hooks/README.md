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
