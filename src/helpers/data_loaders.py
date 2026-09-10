from pathlib import Path

import xarray as xr

from settings import settings
from structs.point import Point

DATA_ROOT = settings.PROJECT_ROOT / "data" / "NZ_domain"


def _test_subdir(period: str) -> str:
    if period == settings.HIST_PERIOD:
        return "historical"

    if period == settings.FUTURE_PERIOD:
        return "end_century"

    raise ValueError(f"Invalid period: {period}")


def predictors_path(period: str, kind: str = "perfect", gcm: str | None = None) -> Path:
    gcm = gcm or settings.GCM_TRAIN

    if period == settings.TRAIN_PERIOD:
        return DATA_ROOT / "train/ESD_pseudo_reality/predictors" / f"{gcm}_{period}.nc"

    return (
        DATA_ROOT
        / "test"
        / _test_subdir(period)
        / "predictors"
        / kind
        / f"{gcm}_{period}.nc"
    )


def _target_path(period: str, gcm: str | None = None) -> Path:
    gcm = gcm or settings.GCM_TRAIN

    if period == settings.TRAIN_PERIOD:
        return (
            DATA_ROOT
            / "train/ESD_pseudo_reality/target"
            / f"pr_tasmax_{gcm}_{period}.nc"
        )

    return (
        DATA_ROOT
        / "test"
        / _test_subdir(period)
        / "target"
        / f"pr_tasmax_{gcm}_{period}.nc"
    )


def load_predictors(
    period: str, kind: str = "perfect", gcm: str | None = None
) -> xr.Dataset:
    """
    Load large-scale predictor fields for a period and GCM.

    Parameters
    ----------
    period : str
        Climate period encoded in the source file name.
    kind : str, optional
        Predictor data type for test periods. Default is ``"perfect"``.
    gcm : str, optional
        Driving GCM name. When omitted, uses the training GCM.

    Returns
    -------
    xarray.Dataset
        Predictor variables with ``time``, ``lat``, and ``lon`` dimensions.
    """
    gcm = gcm or settings.GCM_TRAIN
    return xr.open_dataset(predictors_path(period, kind, gcm))[settings.PRED_VARS]


def load_precip(period: str, gcm: str | None = None) -> xr.DataArray:
    """
    Load high-resolution daily precipitation for a period and GCM.

    Parameters
    ----------
    period : str
        Climate period encoded in the source file name.
    gcm : str, optional
        Driving GCM name. When omitted, uses the training GCM.

    Returns
    -------
    xarray.DataArray
        Precipitation in mm/day with ``time``, ``lat``, and ``lon`` dimensions.
    """
    gcm = gcm or settings.GCM_TRAIN
    return xr.open_dataset(_target_path(period, gcm))["pr"]


def sample_point(da: xr.DataArray, point: Point) -> xr.DataArray:
    """
    Select the nearest grid-cell value or values for a geographic point.

    Parameters
    ----------
    da : xarray.DataArray
        Field with ``lat`` and ``lon`` coordinates.
    point : Point
        Geographic point to sample.

    Returns
    -------
    xarray.DataArray
        Values at the nearest latitude and longitude grid cell.
    """
    return da.sel(lat=point["lat"], lon=point["lon"], method="nearest")
