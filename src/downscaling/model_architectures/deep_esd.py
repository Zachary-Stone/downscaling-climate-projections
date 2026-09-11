"""DeepESD convolutional model architecture."""

import torch
import torch.nn as nn

from config import settings
from downscaling.data.loaders import CordexDataInterface


class DeepESD(nn.Module):
    """Convolutional baseline that maps coarse predictors to grid-cell targets."""

    def __init__(
        self,
        n_predictors: int,
        n_outputs: int,
        in_hw: tuple[int, int],
        filters_last_conv: int = 1,
    ) -> None:
        """
        Initialize the DeepESD convolutional downscaling model.

        Parameters
        ----------
        n_predictors : int
            Number of large-scale predictor channels.
        n_outputs : int
            Number of flattened target-grid cells.
        in_hw : tuple[int, int]
            Height and width of each predictor field.
        filters_last_conv : int, optional
            Number of channels in the final convolution. Default is 1.
        """
        super().__init__()
        self.conv_1 = nn.Conv2d(n_predictors, 50, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(25, filters_last_conv, kernel_size=3, padding=1)
        flat = in_hw[0] * in_hw[1] * filters_last_conv
        self.out = nn.Linear(flat, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produce flattened precipitation predictions for a batch of predictors.

        Parameters
        ----------
        x : torch.Tensor
            Predictor batch with shape ``(batch, variable, lat, lon)``.

        Returns
        -------
        torch.Tensor
            Predictions with shape ``(batch, n_outputs)``.
        """
        x = torch.relu(self.conv_1(x))
        x = torch.relu(self.conv_2(x))
        x = torch.relu(self.conv_3(x))
        x = torch.flatten(x, start_dim=1)
        return self.out(x)


def build_cordex_esd(filters_last_conv: int = 1) -> DeepESD:
    """
    Create a DeepESD model configured for the current predictor grid.

    Parameters
    ----------
    filters_last_conv : int, optional
        Number of channels in the final convolution. Default is 1.

    Returns
    -------
    DeepESD
        Fresh model moved to the configured compute device.
    """
    predictors_train = CordexDataInterface.predictors_train()
    return DeepESD(
        n_predictors=len(settings.PRED_VARS),
        n_outputs=CordexDataInterface.n_gridpoints(),
        in_hw=(predictors_train.sizes["lat"], predictors_train.sizes["lon"]),
        filters_last_conv=filters_last_conv,
    ).to(settings.DEVICE)


if __name__ == "__main__":
    _demo = build_cordex_esd()

    n_params = sum(p.numel() for p in _demo.parameters())
    print(_demo)
    print(f"\nTrainable parameters: {n_params:,}")
