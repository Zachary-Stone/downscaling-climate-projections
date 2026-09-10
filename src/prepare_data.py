import numpy as np
from torch.utils.data import DataLoader

from helpers.data_loaders import load_precip, load_predictors
from settings import settings
from structs.emulation_dataset import EmulationDataset

# Load the training predictors and predictand
# Dataset: 15 vars, (time, lat, lon)
predictors_train = load_predictors(settings.TRAIN_PERIOD)
precip_train = load_precip(settings.TRAIN_PERIOD)  # DataArray: pr (time, lat, lon)

print("Predictors:", dict(predictors_train.sizes))
print(
    "Predictand 'pr':",
    dict(precip_train.sizes),
    "units:",
    precip_train.attrs.get("units"),
)

EXAMPLE_DAY = "1972-07-15"  # 🔧 try other dates within the training period

day_preds = predictors_train.sel(time=EXAMPLE_DAY, method="nearest")
day_pr = precip_train.sel(time=EXAMPLE_DAY, method="nearest")
example_date = str(np.datetime_as_string(day_preds.time.values, unit="D"))

PRED_CMAPS = {
    "u": "RdBu_r",
    "v": "RdBu_r",
    "q": "YlGnBu",
    "t": "coolwarm",
    "z": "viridis",
}
_LONG = {
    "u": "Zonal wind",
    "v": "Meridional wind",
    "q": "Specific humidity",
    "t": "Temperature",
    "z": "Geopotential height",
}
_VARS = ["u", "v", "q", "t", "z"]
_LEVS = [850, 700, 500]

fig, axes = plt.subplots(5, 3, figsize=(10, 16))
for row, v in enumerate(_VARS):
    for col, lev in enumerate(_LEVS):
        var = f"{v}_{lev}"
        ax = axes[row, col]
        plot_field(
            ax,
            day_preds[var],
            title=f"{lev} hPa",
            cmap=PRED_CMAPS[v],
            cbar_label=day_preds[var].attrs.get("units", ""),
            title_fontsize=9,
        )
        if col == 0:
            ax.set_ylabel(_LONG[v], fontsize=9, labelpad=4)
fig.suptitle(f"Predictors (coarse GCM fields) on {example_date}", fontsize=13)
fig.subplots_adjust(hspace=0.55, wspace=0.65, top=0.95)
plt.show()

subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.2), subplot_kw=subplot_kw)
plot_field(
    ax,
    day_pr,
    title=f"Predictand: daily precip on {example_date}",
    cmap="YlGnBu",
    cbar_label="mm/day",
)
plt.show()


def standardization_stats(predictors_ds):
    """Per-field mean and std over time (computed on the TRAIN years only)."""
    return predictors_ds.mean("time"), predictors_ds.std("time")


def predictors_to_array(predictors_ds, mean, std):
    """Standardize and return a float32 array shaped (time, n_vars, lat, lon)."""
    ds_std = (predictors_ds - mean) / std
    arr = ds_std.to_array(dim="variable").transpose("time", "variable", "lat", "lon")
    return arr.values.astype("float32")


def precip_to_array(precip_da):
    """Flatten the spatial grid: (time, lat, lon) -> (time, lat*lon), float32."""
    n_time = precip_da.sizes["time"]
    return precip_da.values.reshape(n_time, -1).astype("float32")


# Build train / validation splits, tensors and DataLoaders
year_start, year_end = settings.TRAIN_PERIOD.split("-")
train_slice: slice = slice(year_start, str(settings.VAL_SPLIT_YEAR - 1))
val_slice: slice = slice(str(settings.VAL_SPLIT_YEAR), year_end)

# Split predictors and precipitation in time
pred_tr = predictors_train.sel(time=train_slice)
pred_va = predictors_train.sel(time=val_slice)
pr_tr = precip_train.sel(time=train_slice)
pr_va = precip_train.sel(time=val_slice)

# Standardization statistics from the TRAIN years only, reused everywhere later
PRED_MEAN, PRED_STD = standardization_stats(pred_tr)

# Target grid (kept so we can turn model output back into maps later)
TARGET_LAT = precip_train["lat"]
TARGET_LON = precip_train["lon"]
N_GRIDPOINTS = TARGET_LAT.size * TARGET_LON.size

x_tr = predictors_to_array(pred_tr, PRED_MEAN, PRED_STD)
x_va = predictors_to_array(pred_va, PRED_MEAN, PRED_STD)
y_tr = precip_to_array(pr_tr)
y_va = precip_to_array(pr_va)

train_loader = DataLoader(
    EmulationDataset(x_tr, y_tr), batch_size=settings.BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    EmulationDataset(x_va, y_va), batch_size=settings.BATCH_SIZE, shuffle=False
)

print(f"Train days: {x_tr.shape[0]}  ({train_slice.start}-{train_slice.stop})")
print(f"Val   days: {x_va.shape[0]}  ({val_slice.start}-{val_slice.stop})")
print(f"Predictor tensor shape (days, vars, lat, lon): {x_tr.shape}")
print(f"Target tensor shape    (days, gridpoints):     {y_tr.shape}")
