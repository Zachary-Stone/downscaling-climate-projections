from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from config import settings
from downscaling.data.plotting import make_map_axes, plot_field
from structs.emulation_dataset import EmulationDataset
from structs.point import Point


def _data_root() -> Path:
    """
    Return the directory containing the extracted NZ CORDEX data.

    Returns
    -------
    pathlib.Path
        Absolute path to the ``NZ_domain`` data directory.
    """
    return settings.PROJECT_ROOT / "data" / "NZ_domain"


def _test_subdir(period: str) -> str:
    if period == settings.HIST_PERIOD:
        return "historical"

    if period == settings.FUTURE_PERIOD:
        return "end_century"

    raise ValueError(f"Invalid period: {period}")


def _predictors_path(
    period: str, kind: str = "perfect", gcm: str | None = None
) -> Path:
    gcm = gcm or settings.GCM_TRAIN

    if period == settings.TRAIN_PERIOD:
        return (
            _data_root() / "train/ESD_pseudo_reality/predictors" / f"{gcm}_{period}.nc"
        )

    return (
        _data_root()
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
            _data_root()
            / "train/ESD_pseudo_reality/target"
            / f"pr_tasmax_{gcm}_{period}.nc"
        )

    return (
        _data_root()
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
    return xr.open_dataset(_predictors_path(period, kind, gcm))[settings.PRED_VARS]


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


def _standardization_stats(predictors_ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Calculate per-field standardization statistics across time.

    Parameters
    ----------
    predictors_ds : xarray.Dataset
        Predictor fields for the period used to fit the statistics.

    Returns
    -------
    tuple[xarray.Dataset, xarray.Dataset]
        Per-field mean and standard deviation over the ``time`` dimension.
    """
    return predictors_ds.mean("time"), predictors_ds.std("time")


def _predictors_to_array(
    predictors_ds: xr.Dataset, mean: xr.Dataset, std: xr.Dataset
) -> np.ndarray:
    """
    Standardize predictor fields and convert them to a float32 array.

    Parameters
    ----------
    predictors_ds : xarray.Dataset
        Predictor fields to standardize.
    mean : xarray.Dataset
        Per-field means used for standardization.
    std : xarray.Dataset
        Per-field standard deviations used for standardization.

    Returns
    -------
    numpy.ndarray
        Standardized values with shape ``(time, variable, lat, lon)``.
    """
    standardized = (predictors_ds - mean) / std
    array = standardized.to_array(dim="variable").transpose(
        "time", "variable", "lat", "lon"
    )
    return array.values.astype("float32")


def _precip_to_array(precip_da: xr.DataArray) -> np.ndarray:
    """
    Flatten a precipitation spatial grid into target vectors.

    Parameters
    ----------
    precip_da : xarray.DataArray
        Precipitation values with ``time``, ``lat``, and ``lon`` dimensions.

    Returns
    -------
    numpy.ndarray
        Float32 values with shape ``(time, lat * lon)``.
    """
    n_time = precip_da.sizes["time"]
    return precip_da.values.reshape(n_time, -1).astype("float32")


class CordexDataInterface:
    """
    Lazily load and cache CORDEX training data and derived values.

    The interface owns the training and validation splits, standardization
    statistics, target-grid metadata, and NumPy entries. Each is constructed
    only on its first use and shared by later callers.
    """

    _predictors_train: xr.Dataset | None = None
    _precip_train: xr.DataArray | None = None
    _pred_train: xr.Dataset | None = None
    _pred_validation: xr.Dataset | None = None
    _precip_train_split: xr.DataArray | None = None
    _precip_validation: xr.DataArray | None = None
    _pred_mean: xr.Dataset | None = None
    _pred_std: xr.Dataset | None = None
    _target_lat: xr.DataArray | None = None
    _target_lon: xr.DataArray | None = None
    _train_entries: tuple[np.ndarray, np.ndarray] | None = None
    _validation_entries: tuple[np.ndarray, np.ndarray] | None = None
    _n_gridpoints: int | None = None

    @classmethod
    def predictors_train(cls) -> xr.Dataset:
        if cls._predictors_train is None:
            cls._predictors_train = load_predictors(settings.TRAIN_PERIOD)
        return cls._predictors_train

    @classmethod
    def precip_train(cls) -> xr.DataArray:
        if cls._precip_train is None:
            cls._precip_train = load_precip(settings.TRAIN_PERIOD)
        return cls._precip_train

    @classmethod
    def train_slice(cls) -> slice:
        year_start, _ = settings.TRAIN_PERIOD.split("-")
        return slice(year_start, str(settings.VAL_SPLIT_YEAR - 1))

    @classmethod
    def validation_slice(cls) -> slice:
        _, year_end = settings.TRAIN_PERIOD.split("-")
        return slice(str(settings.VAL_SPLIT_YEAR), year_end)

    @classmethod
    def train_predictors(cls) -> xr.Dataset:
        if cls._pred_train is None:
            cls._pred_train = cls.predictors_train().sel(time=cls.train_slice())
        return cls._pred_train

    @classmethod
    def validation_predictors(cls) -> xr.Dataset:
        if cls._pred_validation is None:
            cls._pred_validation = cls.predictors_train().sel(
                time=cls.validation_slice()
            )
        return cls._pred_validation

    @classmethod
    def train_precip(cls) -> xr.DataArray:
        if cls._precip_train_split is None:
            cls._precip_train_split = cls.precip_train().sel(time=cls.train_slice())
        return cls._precip_train_split

    @classmethod
    def validation_precip(cls) -> xr.DataArray:
        if cls._precip_validation is None:
            cls._precip_validation = cls.precip_train().sel(time=cls.validation_slice())
        return cls._precip_validation

    @classmethod
    def pred_mean(cls) -> xr.Dataset:
        if cls._pred_mean is None:
            cls._pred_mean, cls._pred_std = _standardization_stats(
                cls.train_predictors()
            )
        return cls._pred_mean

    @classmethod
    def pred_std(cls) -> xr.Dataset:
        if cls._pred_std is None:
            cls._pred_mean, cls._pred_std = _standardization_stats(
                cls.train_predictors()
            )
        return cls._pred_std

    @classmethod
    def target_lat(cls) -> xr.DataArray:
        if cls._target_lat is None:
            cls._target_lat = cls.precip_train()["lat"]
        return cls._target_lat

    @classmethod
    def target_lon(cls) -> xr.DataArray:
        if cls._target_lon is None:
            cls._target_lon = cls.precip_train()["lon"]
        return cls._target_lon

    @classmethod
    def train_entries(cls) -> tuple[np.ndarray, np.ndarray]:
        if cls._train_entries is None:
            cls._train_entries = (
                _predictors_to_array(
                    cls.train_predictors(), cls.pred_mean(), cls.pred_std()
                ),
                _precip_to_array(cls.train_precip()),
            )
        return cls._train_entries

    @classmethod
    def validation_entries(cls) -> tuple[np.ndarray, np.ndarray]:
        if cls._validation_entries is None:
            cls._validation_entries = (
                _predictors_to_array(
                    cls.validation_predictors(), cls.pred_mean(), cls.pred_std()
                ),
                _precip_to_array(cls.validation_precip()),
            )
        return cls._validation_entries

    @classmethod
    def n_gridpoints(cls) -> int:
        if cls._n_gridpoints is None:
            cls._n_gridpoints = cls.target_lat().size * cls.target_lon().size
        return cls._n_gridpoints


def _get_train_loader() -> DataLoader:
    """
    Create a data loader for the cached training entries.

    Returns
    -------
    torch.utils.data.DataLoader
        Shuffled batches of standardized predictors and precipitation targets.
    """
    x_train, y_train = CordexDataInterface.train_entries()
    return DataLoader(
        EmulationDataset(x_train, y_train), batch_size=settings.BATCH_SIZE, shuffle=True
    )


def _get_val_loader() -> DataLoader:
    """
    Create a data loader for the cached validation entries.

    Returns
    -------
    torch.utils.data.DataLoader
        Ordered batches of standardized predictors and precipitation targets.
    """
    x_validation, y_validation = CordexDataInterface.validation_entries()
    return DataLoader(
        EmulationDataset(x_validation, y_validation),
        batch_size=settings.BATCH_SIZE,
        shuffle=False,
    )


def _plot_training_example(example_day: str = "1972-07-15") -> None:
    """
    Plot coarse predictors and high-resolution precipitation for one day.

    Parameters
    ----------
    example_day : str, optional
        Requested date in the training period. The nearest available date is
        plotted. Default is ``"1972-07-15"``.

    Returns
    -------
    None
        This function displays the predictor and precipitation figures.
    """

    predictors = CordexDataInterface.predictors_train()
    precipitation = CordexDataInterface.precip_train()
    day_predictors = predictors.sel(time=example_day, method="nearest")
    day_precipitation = precipitation.sel(time=example_day, method="nearest")
    date = str(np.datetime_as_string(day_predictors.time.values, unit="D"))
    colormaps = {
        "u": "RdBu_r",
        "v": "RdBu_r",
        "q": "YlGnBu",
        "t": "coolwarm",
        "z": "viridis",
    }
    names = {
        "u": "Zonal wind",
        "v": "Meridional wind",
        "q": "Specific humidity",
        "t": "Temperature",
        "z": "Geopotential height",
    }

    fig, axes = plt.subplots(5, 3, figsize=(10, 16))
    for row, variable in enumerate(["u", "v", "q", "t", "z"]):
        for column, level in enumerate([850, 700, 500]):
            field_name = f"{variable}_{level}"
            ax = axes[row, column]
            plot_field(
                ax,
                day_predictors[field_name],
                title=f"{level} hPa",
                cmap=colormaps[variable],
                cbar_label=day_predictors[field_name].attrs.get("units", ""),
                title_fontsize=9,
            )
            if column == 0:
                ax.set_ylabel(names[variable], fontsize=9, labelpad=4)
    fig.suptitle(f"Predictors (coarse GCM fields) on {date}", fontsize=13)
    fig.subplots_adjust(hspace=0.55, wspace=0.65, top=0.95)
    plt.show()

    _, axes = make_map_axes(ncols=1, figsize=(6.5, 5.2))
    plot_field(
        axes[0],
        day_precipitation,
        title=f"Predictand: daily precip on {date}",
        cmap="YlGnBu",
        cbar_label="mm/day",
    )
    plt.show()


if __name__ == "__main__":
    x_train, y_train = CordexDataInterface.train_entries()
    x_validation, y_validation = CordexDataInterface.validation_entries()
    train_slice = CordexDataInterface.train_slice()
    validation_slice = CordexDataInterface.validation_slice()
    print(f"Train days: {x_train.shape[0]}  ({train_slice.start}-{train_slice.stop})")
    print(
        f"Val   days: {x_validation.shape[0]}  "
        f"({validation_slice.start}-{validation_slice.stop})"
    )
    print(f"Predictor tensor shape (days, vars, lat, lon): {x_train.shape}")
    print(f"Target tensor shape    (days, gridpoints):     {y_train.shape}")
    _plot_training_example()
