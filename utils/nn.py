import torch.nn as nn

def init_module_weights(m: nn.Module, std: float = 0.02) -> None:
    """
    Initialize weights for common layer types using truncated normal distribution.
    Adapted from eb_jepa.nn_utils.init_module_weights.

    Use via: module.apply(lambda m: init_module_weights(m, std=0.02))

    Args:
        m: PyTorch module to initialize
        std: Standard deviation for truncated normal initialization (default: 0.02)
    """
    if isinstance(
        m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    ):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
