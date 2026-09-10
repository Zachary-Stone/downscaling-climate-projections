from pathlib import Path
from typing import TypedDict

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from settings import settings


class Point(TypedDict):
    """
    A named geographic point used for sampling and map annotations.

    Parameters
    ----------
    name : str
        Human-readable point name.
    lat : float
        Point latitude in degrees north.
    lon : float
        Point longitude in degrees east.
    """

    name: str
    lat: float
    lon: float


def _test_subdir(period: str) -> str:
    return {settings.HIST_PERIOD: "historical", settings.FUTURE_PERIOD: "end_century"}[
        period
    ]


def predictors_path(period: str, kind: str = "perfect", gcm: str | None = None) -> Path:
    gcm = gcm or settings.GCM_TRAIN

    if period == settings.TRAIN_PERIOD:
        return (
            settings.DATA_ROOT
            / "train/ESD_pseudo_reality/predictors"
            / f"{gcm}_{period}.nc"
        )

    return (
        settings.DATA_ROOT
        / "test"
        / _test_subdir(period)
        / "predictors"
        / kind
        / f"{gcm}_{period}.nc"
    )


def target_path(period: str, gcm: str | None = None) -> Path:
    gcm = gcm or settings.GCM_TRAIN

    if period == settings.TRAIN_PERIOD:
        return (
            settings.DATA_ROOT
            / "train/ESD_pseudo_reality/target"
            / f"pr_tasmax_{gcm}_{period}.nc"
        )

    return (
        settings.DATA_ROOT
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
    return xr.open_dataset(target_path(period, gcm))["pr"]


def make_map_axes(
    ncols: int = 1, figsize: tuple[float, float] | None = None
) -> tuple[Figure, np.ndarray]:
    kw = {"subplot_kw": {"projection": ccrs.PlateCarree()}}
    fig, axes = plt.subplots(1, ncols, figsize=figsize or (5.2 * ncols, 4.8), **kw)
    axes = np.atleast_1d(axes)
    return fig, axes


def plot_field(
    ax: Axes,
    da: xr.DataArray,
    title: str = "",
    cmap: str | Colormap = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    points: list[Point] | None = None,
    cbar_label: str = "",
    mark_color: str = "red",
    title_fontsize: float = 11,
) -> QuadMesh:
    """
    Plot a two-dimensional field, optionally marking geographic points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cartesian or Cartopy geographic axes on which to plot the field.
    da : xarray.DataArray
        Two-dimensional field with ``lat`` and ``lon`` coordinates.
    title : str, optional
        Axes title. Default is ``""``.
    cmap : str or matplotlib.colors.Colormap, optional
        Colormap used for the field. Default is ``"viridis"``.
    vmin, vmax : float, optional
        Lower and upper bounds of the colormap normalization.
    points : list of Point, optional
        Named geographic points to scatter and label.
    cbar_label : str, optional
        Colorbar label. Default is ``""``.
    mark_color : str, optional
        Marker fill color. Default is ``"red"``.
    title_fontsize : float, optional
        Font size for the title. Default is 11.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The mesh artist returned by ``Axes.pcolormesh``.
    """
    is_geo = hasattr(ax, "projection")
    geo = {"transform": ccrs.PlateCarree()} if is_geo else {}
    mesh = ax.pcolormesh(
        da["lon"], da["lat"], da, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", **geo
    )
    lon_min, lon_max = float(da.lon.min()), float(da.lon.max())
    lat_min, lat_max = float(da.lat.min()), float(da.lat.max())
    if is_geo:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        try:
            ax.coastlines(resolution="10m", linewidth=0.7, color="black")
        except Exception:
            pass
    else:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=title_fontsize)
    if points:
        for p in points:
            ax.scatter(
                p["lon"],
                p["lat"],
                s=45,
                color=mark_color,
                edgecolor="black",
                zorder=5,
                **geo,
            )
            ax.text(
                p["lon"] + 0.12,
                p["lat"],
                p["name"].split(" (")[0],
                fontsize=8,
                zorder=6,
                **geo,
            )
    plt.colorbar(mesh, ax=ax, shrink=0.75, pad=0.02, label=cbar_label)
    return mesh


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
