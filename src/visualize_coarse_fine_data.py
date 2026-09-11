import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from config import settings
from downscaling.data.loaders import load_precip, sample_point
from downscaling.data.plotting import make_map_axes, plot_field

# The two locations (NZ South Island)
# 🔧 Try other pairs of points once you understand the story.
POINT_WEST = {
    "name": "West Coast (Hokitika)",
    "lat": -42.72,
    "lon": 170.97,
}
POINT_EAST = {
    "name": "Southern Alps (mountain crest)",
    "lat": -42.95,
    "lon": 171.30,
}
POINTS = [POINT_WEST, POINT_EAST]
COARSEN_FACTOR = 16


def _upscale_to_coarse(da: xr.DataArray, factor: int | None = None) -> xr.DataArray:
    """
    Average a latitude-longitude field onto a coarser grid.

    Parameters
    ----------
    da : xarray.DataArray
        Field with ``lat`` and ``lon`` dimensions to aggregate.
    factor : int, optional
        Number of grid cells to average along each spatial dimension. When
        omitted, uses ``settings.COARSEN_FACTOR``.

    Returns
    -------
    xarray.DataArray
        Field coarsened along the ``lat`` and ``lon`` dimensions. Partial
        cells at the spatial boundaries are discarded.
    """
    factor = factor or COARSEN_FACTOR
    return da.coarsen(lat=factor, lon=factor, boundary="trim").mean()


def _plot_mean_daily_precip() -> xr.DataArray:
    """
    Plot historical mean daily precipitation and report values at key points.

    Returns
    -------
    xarray.DataArray
        Historical mean daily precipitation with ``lat`` and ``lon``
        dimensions.
    """
    pr_hist = load_precip(settings.HIST_PERIOD)
    pr_hist_clim = pr_hist.mean("time").compute()  # mean over the 20-year period

    _, axes = make_map_axes(ncols=1, figsize=(6, 5.5))
    plot_field(
        axes[0],
        pr_hist_clim,
        title=f"Mean daily precipitation, {settings.HIST_PERIOD} (10 km)",
        cmap="YlGnBu",
        vmin=0,
        vmax=30,
        points=POINTS,
        cbar_label="mm/day",
    )
    plt.show()

    for p in POINTS:
        print(f"  {p['name']:>28}: {float(sample_point(pr_hist_clim, p)):5.2f} mm/day")

    return pr_hist_clim


def _plot_upscaled_hist_precip(pr_hist_clim: xr.DataArray) -> xr.DataArray:
    """
    Compare historical precipitation at high and coarse spatial resolutions.

    Parameters
    ----------
    pr_hist_clim : xarray.DataArray
        Historical mean daily precipitation with ``lat`` and ``lon``
        dimensions.

    Returns
    -------
    xarray.DataArray
        ``pr_hist_clim`` aggregated to the coarse grid.
    """
    # TODO: make ruff allow linebreaks under func def

    pr_hist_clim_coarse = _upscale_to_coarse(pr_hist_clim)

    _, axes = make_map_axes(ncols=2, figsize=(11, 4.8))
    plot_field(
        axes[0],
        pr_hist_clim,
        title="High resolution (~10 km)",
        cmap="YlGnBu",
        vmin=0,
        vmax=30,
        points=POINTS,
        cbar_label="mm/day",
    )
    plot_field(
        axes[1],
        pr_hist_clim_coarse,
        title="Coarse / GCM-like (~2deg)",
        cmap="YlGnBu",
        vmin=0,
        vmax=30,
        points=POINTS,
        cbar_label="mm/day",
    )
    plt.suptitle(f"Mean daily precipitation, {settings.HIST_PERIOD}", y=1.02)
    plt.show()

    print("Mean precipitation at each location (mm/day):")
    print(f"  {'location':>28} | high-res | coarse")
    for p in POINTS:
        hi = float(sample_point(pr_hist_clim, p))
        lo = float(sample_point(pr_hist_clim_coarse, p))
        print(f"  {p['name']:>28} | {hi:7.2f}  | {lo:6.2f}")

    return pr_hist_clim_coarse


def _plot_forecasted_precip(
    pr_hist_clim: xr.DataArray, pr_hist_clim_coarse: xr.DataArray
) -> None:
    """
    Plot projected precipitation changes at high and coarse resolutions.

    Parameters
    ----------
    pr_hist_clim : xarray.DataArray
        Historical mean daily precipitation with ``lat`` and ``lon``
        dimensions.
    pr_hist_clim_coarse : xarray.DataArray
        Coarse-grid historical mean daily precipitation aligned with the
        coarse future field.

    Returns
    -------
    None
        This function displays maps and a point-based bar chart.
    """
    pr_future = load_precip(settings.FUTURE_PERIOD)  # ACCESS-CM2, 2080-2099
    pr_future_clim = pr_future.mean("time").compute()

    # Change signal = future mean - historical mean.
    change_hi = pr_future_clim - pr_hist_clim
    change_coarse = _upscale_to_coarse(pr_future_clim) - pr_hist_clim_coarse

    vlim = float(np.nanmax(np.abs(change_hi))) * 0.8
    _, axes = make_map_axes(ncols=2, figsize=(11, 4.8))
    plot_field(
        axes[0],
        change_hi,
        title="Change, high resolution (~10 km)",
        cmap="BrBG",
        vmin=-vlim,
        vmax=vlim,
        points=POINTS,
        cbar_label="mm/day",
    )
    plot_field(
        axes[1],
        change_coarse,
        title="Change, coarse / GCM-like (~2deg)",
        cmap="BrBG",
        vmin=-vlim,
        vmax=vlim,
        points=POINTS,
        cbar_label="mm/day",
    )
    plt.suptitle(
        f"Precipitation change, {settings.FUTURE_PERIOD} minus {settings.HIST_PERIOD}",
        y=1.02,
    )
    plt.show()

    # Bar chart of the change at the two locations, at each resolution
    labels = [p["name"].split(" (")[0] for p in POINTS]
    chg_hi_pts = [float(sample_point(change_hi, p)) for p in POINTS]
    chg_lo_pts = [float(sample_point(change_coarse, p)) for p in POINTS]
    x = np.arange(len(POINTS))
    _, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, chg_hi_pts, width=0.4, label="high resolution")
    ax.bar(x + 0.2, chg_lo_pts, width=0.4, label="coarse / GCM-like")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Precip. change (mm/day)")
    ax.set_title("Future precipitation change at the two locations")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    pr_hist_clim = _plot_mean_daily_precip()
    pr_hist_clim_coarse = _plot_upscaled_hist_precip(pr_hist_clim)
    _plot_forecasted_precip(pr_hist_clim, pr_hist_clim_coarse)
