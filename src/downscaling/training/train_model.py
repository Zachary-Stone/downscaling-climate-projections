"""Train downscaling models and convert predictions to precipitation fields."""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader

from config import settings
from downscaling.data.loaders import CordexDataInterface, predictors_to_array

TensorBatch = tuple[torch.Tensor, torch.Tensor]
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def evaluate_loss(
    model: nn.Module, loader: DataLoader[TensorBatch], loss_fn: LossFunction
) -> float:
    """
    Calculate average loss for a model over a data loader.

    Parameters
    ----------
    model : torch.nn.Module
        Model that maps predictor batches to target batches.
    loader : torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]
        Batches of predictors and target precipitation vectors.
    loss_fn : collections.abc.Callable
        Loss function applied to predicted and target tensors.

    Returns
    -------
    float
        Sample-weighted mean loss across the loader.
    """
    model.eval()
    total, n = 0.0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(settings.DEVICE), yb.to(settings.DEVICE)
            total += loss_fn(model(xb), yb).item() * xb.size(0)
            n += xb.size(0)

    return total / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader[TensorBatch],
    val_loader: DataLoader[TensorBatch],
    num_epochs: int = settings.NUM_EPOCHS,
    lr: float = settings.LEARNING_RATE,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Train a model with Adam optimization and mean-squared error.

    Parameters
    ----------
    model : torch.nn.Module
        Model that maps ``(batch, variable, lat, lon)`` tensors to flattened
        target-grid vectors.
    train_loader : torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]
        Shuffled predictor and target batches used for optimization.
    val_loader : torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]
        Predictor and target batches used for validation.
    num_epochs : int, optional
        Number of optimization epochs. Default is ``NUM_EPOCHS``.
    lr : float, optional
        Adam learning rate. Default is ``LEARNING_RATE``.
    verbose : bool, optional
        Whether to print periodic loss updates. Default is True.

    Returns
    -------
    dict[str, list[float]]
        Training and validation mean-squared-error histories, keyed by
        ``"train"`` and ``"val"``.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(settings.DEVICE), yb.to(settings.DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)

        history["train"].append(running / n)
        history["val"].append(evaluate_loss(model, val_loader, loss_fn))

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == num_epochs):
            print(
                f"  epoch {epoch:3d}/{num_epochs}  "
                f"train MSE {history['train'][-1]:.3f}  "
                f"val MSE {history['val'][-1]:.3f}"
            )

    return history


def downscale(
    model: nn.Module,
    predictors_ds: xr.Dataset,
    mean: xr.Dataset | None = None,
    std: xr.Dataset | None = None,
    batch: int = 512,
) -> xr.DataArray:
    """
    Apply a trained model to predictor fields.

    Predictors are standardized with training statistics because only those
    statistics are available when producing a projection.

    Parameters
    ----------
    model : torch.nn.Module
        Model that returns flattened precipitation predictions.
    predictors_ds : xarray.Dataset
        Predictor fields with ``time``, ``lat``, and ``lon`` dimensions.
    mean : xarray.Dataset, optional
        Predictor means used for standardization. When omitted, uses the
        training-period means.
    std : xarray.Dataset, optional
        Predictor standard deviations used for standardization. When omitted,
        uses the training-period standard deviations.
    batch : int, optional
        Number of time steps evaluated per model call. Default is 512.

    Returns
    -------
    xarray.DataArray
        Non-negative precipitation predictions with dimensions ``(time, lat,
        lon)`` and units of mm/day.
    """
    if mean is None:
        mean = CordexDataInterface.pred_mean()
    if std is None:
        std = CordexDataInterface.pred_std()
    x = predictors_to_array(predictors_ds, mean, std)

    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i : i + batch]).to(settings.DEVICE)
            chunks.append(model(xb).cpu().numpy())
    pred = np.concatenate(chunks, axis=0)
    pred = pred.reshape(
        pred.shape[0],
        CordexDataInterface.target_lat().size,
        CordexDataInterface.target_lon().size,
    )

    da = xr.DataArray(
        pred,
        dims=("time", "lat", "lon"),
        coords={
            "time": predictors_ds["time"],
            "lat": CordexDataInterface.target_lat(),
            "lon": CordexDataInterface.target_lon(),
        },
        name="pr",
    )

    # Precipitation cannot be negative, and MSE training can produce small negatives.
    da = da.clip(min=0.0)
    da.attrs["units"] = "mm day-1"
    return da
