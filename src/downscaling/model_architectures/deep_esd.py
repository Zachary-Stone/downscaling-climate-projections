import torch
import torch.nn as nn

from config import settings
from downscaling.data.loaders import load_predictors


class DeepESD(nn.Module):
    """Convolutional downscaling baseline.

    Args:
        n_predictors: number of input channels (large-scale fields).
        n_outputs:    number of output grid cells (flattened 128*128).
        in_hw:        spatial size (H, W) of the predictor grid, e.g. (16, 16).
        filters_last_conv: channels in the third conv layer.
    """

    def __init__(self, n_predictors, n_outputs, in_hw, filters_last_conv=1):
        super().__init__()
        self.conv_1 = nn.Conv2d(n_predictors, 50, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(25, filters_last_conv, kernel_size=3, padding=1)
        flat = in_hw[0] * in_hw[1] * filters_last_conv
        self.out = nn.Linear(flat, n_outputs)

    def forward(self, x):
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.flatten(x, start_dim=1)
        return self.out(x)


def cordex_esd():
    """Create a fresh DeepESD configured for the current data shapes."""
    predictors_train = load_predictors(settings.TRAIN_PERIOD)
    return DeepESD(
        n_predictors=len(settings.PRED_VARS),
        n_outputs=settings.N_GRIDPOINTS,
        in_hw=(predictors_train.sizes["lat"], predictors_train.sizes["lon"]),
        filters_last_conv=settings.FILTERS_LAST_CONV,
    ).to(settings.DEVICE)


if __name__ == "__main__":
    _demo = cordex_esd()

    n_params = sum(p.numel() for p in _demo.parameters())
    print(_demo)
    print(f"\nTrainable parameters: {n_params:,}")
