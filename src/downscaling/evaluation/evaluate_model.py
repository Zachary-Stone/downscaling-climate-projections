"""Calculate spatial precipitation diagnostics for model evaluation."""

from collections.abc import Callable

import numpy as np
import xarray as xr

from config import settings

IndexFunction = Callable[[xr.DataArray], xr.DataArray]


def idx_mean(da: xr.DataArray) -> xr.DataArray:
    """
    Calculate the temporal mean precipitation field.

    Parameters
    ----------
    da : xarray.DataArray
        Precipitation data with a ``time`` dimension.

    Returns
    -------
    xarray.DataArray
        Mean precipitation at each grid cell.
    """
    return da.mean("time")


def idx_quantile(da: xr.DataArray, q: float) -> xr.DataArray:
    """
    Calculate a temporal precipitation quantile at each grid cell.

    Parameters
    ----------
    da : xarray.DataArray
        Precipitation data with a ``time`` dimension.
    q : float
        Quantile to calculate, in the closed interval from 0 to 1.

    Returns
    -------
    xarray.DataArray
        Spatial quantile field without the scalar ``quantile`` coordinate.
    """
    out = da.quantile(q, dim="time")
    return out.drop_vars("quantile", errors="ignore")


def idx_p98(da: xr.DataArray) -> xr.DataArray:
    """
    Calculate the 98th-percentile precipitation field.

    Parameters
    ----------
    da : xarray.DataArray
        Precipitation data with a ``time`` dimension.

    Returns
    -------
    xarray.DataArray
        98th-percentile precipitation at each grid cell.
    """
    return idx_quantile(da, 0.98)


def idx_sdii(
    da: xr.DataArray, wet_threshold: float = settings.WET_DAY_THRESHOLD
) -> xr.DataArray:
    """
    Calculate mean precipitation on wet days at each grid cell.

    Parameters
    ----------
    da : xarray.DataArray
        Precipitation data with a ``time`` dimension.
    wet_threshold : float, optional
        Minimum precipitation that defines a wet day, in mm/day. Default is
        the configured wet-day threshold.

    Returns
    -------
    xarray.DataArray
        Mean precipitation for days at or above ``wet_threshold``.
    """
    return da.where(da >= wet_threshold).mean("time")


def idx_rx1day(da: xr.DataArray) -> xr.DataArray:
    """
    Calculate the mean annual maximum one-day precipitation.

    Parameters
    ----------
    da : xarray.DataArray
        Precipitation data with a ``time`` dimension.

    Returns
    -------
    xarray.DataArray
        Mean across years of the annual maximum daily precipitation.
    """
    return da.groupby("time.year").max("time").mean("year")


def rmse_map(obs: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """
    Calculate temporal root-mean-square error at each grid cell.

    Parameters
    ----------
    obs : xarray.DataArray
        Reference precipitation data with a ``time`` dimension.
    pred : xarray.DataArray
        Predicted precipitation data aligned with ``obs``.

    Returns
    -------
    xarray.DataArray
        Root-mean-square error at each grid cell.
    """
    return np.sqrt(((pred - obs) ** 2).mean("time"))


def index_bias(
    obs: xr.DataArray, pred: xr.DataArray, index_fn: IndexFunction
) -> xr.DataArray:
    """
    Calculate model-minus-reference bias for a spatial index.

    Parameters
    ----------
    obs : xarray.DataArray
        Reference precipitation data.
    pred : xarray.DataArray
        Predicted precipitation data aligned with ``obs``.
    index_fn : collections.abc.Callable
        Function that reduces a precipitation field to a spatial index.

    Returns
    -------
    xarray.DataArray
        Spatial index bias calculated as prediction minus reference.
    """
    return index_fn(pred) - index_fn(obs)


def align_time(pred: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
    """
    Assign a reference time coordinate to predictions of equal length.

    Parameters
    ----------
    pred : xarray.DataArray
        Predicted precipitation with a ``time`` dimension.
    obs : xarray.DataArray
        Reference precipitation with a ``time`` dimension.

    Returns
    -------
    xarray.DataArray
        ``pred`` with the time coordinates from ``obs``.

    Raises
    ------
    AssertionError
        If ``pred`` and ``obs`` contain different numbers of time steps.
    """
    assert pred.sizes["time"] == obs.sizes["time"], (
        "prediction/reference length mismatch"
    )
    return pred.assign_coords(time=obs["time"].values)


def summarize(obs: xr.DataArray, pred: xr.DataArray) -> dict[str, float]:
    """
    Summarize spatially averaged precipitation diagnostics.

    Index biases are averaged in absolute value over the domain, so that
    overestimation in one area and underestimation in another do not cancel.

    Parameters
    ----------
    obs : xarray.DataArray
        Reference precipitation data with a ``time`` dimension.
    pred : xarray.DataArray
        Predicted precipitation data aligned with ``obs``.

    Returns
    -------
    dict[str, float]
        Root-mean-square error and mean absolute bias for each configured
        index, in mm/day.
    """
    # summary statistic order
    sum_stat_order = {
        "mean": idx_mean,
        "SDII": idx_sdii,
        "P98": idx_p98,
        "RX1day": idx_rx1day,
    }

    row = {"RMSE": float(rmse_map(obs, pred).mean())}
    for name, fn in sum_stat_order.items():
        row[f"bias_{name}"] = float(np.abs(index_bias(obs, pred, fn)).mean())
    return row
