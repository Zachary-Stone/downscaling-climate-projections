import numpy as np
import torch
from torch.utils.data import Dataset


class EmulationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Expose predictor arrays and target arrays as float32 tensor samples.

    Parameters
    ----------
    x : numpy.ndarray
        Predictor array whose leading dimension indexes samples.
    y : numpy.ndarray
        Target array with the same number of samples as ``x``.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize the dataset from predictor and target arrays.

        Parameters
        ----------
        x : numpy.ndarray
            Predictor array whose leading dimension indexes samples.
        y : numpy.ndarray
            Target array with the same number of samples as ``x``.
        """
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            Number of predictor-target sample pairs.
        """
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one predictor-target pair.

        Parameters
        ----------
        idx : int
            Zero-based sample index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Float32 predictor tensor and corresponding target tensor.
        """
        return self.x[idx], self.y[idx]
