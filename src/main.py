"""Run the DeepESD training and historical evaluation workflow."""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from codecarbon import EmissionsTracker

from config import settings
from downscaling.data.loaders import (
    get_train_loader,
    get_val_loader,
    load_precip,
    load_predictors,
)
from downscaling.data.plotting import make_map_axes, plot_field
from downscaling.evaluation.evaluate_model import (
    align_time,
    idx_mean,
    idx_p98,
    idx_rx1day,
    idx_sdii,
    index_bias,
    rmse_map,
    summarize,
)
from downscaling.model_architectures.deep_esd import build_cordex_esd
from downscaling.training.train_model import downscale, train_model
from visualize_coarse_fine_data import POINTS

MODELS_DIR = settings.PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "deepesd_pr_nz.pt"
CODECARBON_DIR = settings.PROJECT_ROOT / "code_carbon"


def _plot_loss_history(history: dict[str, list[float]]) -> None:
    """
    Plot training and validation loss by epoch.

    Parameters
    ----------
    history : dict[str, list[float]]
        Mean-squared-error histories produced by ``train_model``.

    Returns
    -------
    None
        This function displays a Matplotlib figure.
    """
    _, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="training")
    ax.plot(epochs, history["val"], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (mm/day)$^2$")
    ax.set_title("Training and validation loss")
    ax.legend()
    plt.show()


def _plot_index_comparison(obs: xr.DataArray, pred: xr.DataArray) -> None:
    """
    Plot reference and downscaled precipitation indices on shared color scales.

    Parameters
    ----------
    obs : xarray.DataArray
        Reference precipitation with a ``time`` dimension.
    pred : xarray.DataArray
        Downscaled precipitation aligned to ``obs``.

    Returns
    -------
    None
        This function displays a Matplotlib figure.
    """
    compare_indices = {
        "Mean (climatology)": idx_mean,
        "SDII": idx_sdii,
        "P98": idx_p98,
        "RX1day": idx_rx1day,
    }
    ncols = len(compare_indices)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(4.2 * ncols, 8.4),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for col, (name, index_fn) in enumerate(compare_indices.items()):
        truth_field = index_fn(obs)
        pred_field = index_fn(pred)
        vmax = float(np.nanmax([truth_field.max(), pred_field.max()]))
        plot_field(
            axes[0, col],
            truth_field,
            title=f"{name} - truth",
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax,
            points=POINTS,
            cbar_label="mm/day",
            title_fontsize=10,
        )
        plot_field(
            axes[1, col],
            pred_field,
            title=f"{name} - DeepESD",
            cmap="YlGnBu",
            vmin=0,
            vmax=vmax,
            points=POINTS,
            cbar_label="mm/day",
            title_fontsize=10,
        )
    fig.suptitle(
        f"Climatology and extremes, {settings.HIST_PERIOD}: "
        "pseudo-reality (top) vs DeepESD (bottom)",
        y=1.01,
    )
    fig.tight_layout()
    plt.show()


def _plot_error_maps(obs: xr.DataArray, pred: xr.DataArray) -> None:
    """
    Plot downscaling RMSE and selected spatial-index biases.

    Parameters
    ----------
    obs : xarray.DataArray
        Reference precipitation with a ``time`` dimension.
    pred : xarray.DataArray
        Downscaled precipitation aligned to ``obs``.

    Returns
    -------
    None
        This function displays a Matplotlib figure.
    """
    rmse_field = rmse_map(obs, pred)
    bias_fields = [
        (index_bias(obs, pred, idx_mean), "mean"),
        (index_bias(obs, pred, idx_p98), "P98"),
        (index_bias(obs, pred, idx_rx1day), "RX1day"),
    ]

    _, axes = make_map_axes(ncols=4, figsize=(18, 4.4))
    plot_field(
        axes[0],
        rmse_field,
        title="RMSE",
        cmap="magma_r",
        points=POINTS,
        cbar_label="mm/day",
    )
    for ax, (field, name) in zip(axes[1:], bias_fields, strict=True):
        vlim = float(np.nanmax(np.abs(field))) * 0.8
        plot_field(
            ax,
            field,
            title=f"Bias of {name} (model - truth)",
            cmap="RdBu",
            vmin=-vlim,
            vmax=vlim,
            points=POINTS,
            cbar_label="mm/day",
        )
    plt.show()


def main() -> pd.DataFrame:
    """
    Train DeepESD, evaluate historical downscaling, and show diagnostics.

    Returns
    -------
    pandas.DataFrame
        One-row table of spatially averaged historical-test diagnostics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CODECARBON_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = get_train_loader()
    val_loader = get_val_loader()
    model = build_cordex_esd()

    print(
        "Training DeepESD from scratch for "
        f"{settings.NUM_EPOCHS} epochs on {settings.DEVICE} ..."
    )

    tracker = EmissionsTracker(
        project_name="deepesd_pr_nz_training",
        output_dir=str(CODECARBON_DIR),
        output_file="deepesd_training_emissions.csv",
        log_level="error",
    )
    tracker.start()
    history = train_model(model, train_loader, val_loader)
    emissions_kg = tracker.stop()

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    if emissions_kg is not None:
        print(
            f"CodeCarbon: training emissions ≈ {emissions_kg:.6f} kg CO₂eq "
            f"(details in {CODECARBON_DIR / 'deepesd_training_emissions.csv'})"
        )
    else:
        print("CodeCarbon: emissions estimate unavailable in this environment.")

    model.eval()

    _plot_loss_history(history)

    predictors_hist = load_predictors(
        settings.HIST_PERIOD, kind="perfect", gcm=settings.GCM_TRAIN
    )
    obs_hist = load_precip(settings.HIST_PERIOD, gcm=settings.GCM_TRAIN).load()

    pr_pred_hist = align_time(downscale(model, predictors_hist), obs_hist)

    summary = summarize(obs_hist, pr_pred_hist)
    summary_df = pd.DataFrame([summary], index=["DeepESD (perfect predictors)"]).round(
        3
    )

    print(
        "Spatially-averaged diagnostics on the "
        f"{settings.HIST_PERIOD} test period (mm/day):"
    )
    print(summary_df)

    _plot_index_comparison(obs_hist, pr_pred_hist)
    _plot_error_maps(obs_hist, pr_pred_hist)
    return summary_df


if __name__ == "__main__":
    main()
