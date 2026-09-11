def idx_mean(da):
    return da.mean("time")


def idx_quantile(da, q):
    out = da.quantile(q, dim="time")
    return out.drop_vars("quantile", errors="ignore")


def idx_p98(da):
    return idx_quantile(da, 0.98)


def idx_sdii(da, wet_threshold=WET_DAY_THRESHOLD):
    """Mean precipitation on wet days (>= threshold)."""
    return da.where(da >= wet_threshold).mean("time")


def idx_rx1day(da):
    """Mean across years of the annual maximum 1-day precipitation."""
    return da.groupby("time.year").max("time").mean("year")


def rmse_map(obs, pred):
    return np.sqrt(((pred - obs) ** 2).mean("time"))


def index_bias(obs, pred, index_fn):
    """Spatial map of (model - truth) for a given index."""
    return index_fn(pred) - index_fn(obs)


# Indices we report, in display order.
INDEX_FUNCS = {"mean": idx_mean, "SDII": idx_sdii, "P98": idx_p98, "RX1day": idx_rx1day}


def summarize(obs, pred):
    """Return a one-row summary of spatially-averaged diagnostics (mm/day).

    Index biases are averaged in absolute value over the domain, so that
    overestimation in one area and underestimation in another do not cancel.
    """
    row = {"RMSE": float(rmse_map(obs, pred).mean())}
    for name, fn in INDEX_FUNCS.items():
        row[f"bias_{name}"] = float(np.abs(index_bias(obs, pred, fn)).mean())
    return row


pred_pred = load_predictors(HIST_PERIOD, kind="perfect", gcm=GCM_TRAIN)
obs_hist = load_precip(HIST_PERIOD, gcm=GCM_TRAIN).load()

pr_pred_hist = align_time(downscale(model, pred_pred), obs_hist)

summary = summarize(obs_hist, pr_pred_hist)
summary_df = pd.DataFrame([summary], index=["DeepESD (perfect predictors)"]).round(3)

print("Spatially-averaged diagnostics on the 1981-2000 test period (mm/day):")
summary_df

COMPARE_INDICES = {
    "Mean (climatology)": idx_mean,
    "SDII": idx_sdii,
    "P98": idx_p98,
    "RX1day": idx_rx1day,
}

ncols = len(COMPARE_INDICES)
subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
fig, axes = plt.subplots(2, ncols, figsize=(4.2 * ncols, 8.4), subplot_kw=subplot_kw)

for col, (name, fn) in enumerate(COMPARE_INDICES.items()):
    truth_field = fn(obs_hist)
    pred_field = fn(pr_pred_hist)
    vmax = float(max(truth_field.max(), pred_field.max()))  # shared scale per column
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
    f"Climatology and extremes, {HIST_PERIOD}: pseudo-reality (top) vs DeepESD (bottom)",
    y=1.01,
)
fig.tight_layout()
plt.show()

# Where are the errors? Maps of RMSE and index biases
rmse_field = rmse_map(obs_hist, pr_pred_hist)
bias_mean_field = index_bias(obs_hist, pr_pred_hist, idx_mean)
bias_p98_field = index_bias(obs_hist, pr_pred_hist, idx_p98)
bias_rx1_field = index_bias(obs_hist, pr_pred_hist, idx_rx1day)

fig, axes = make_map_axes(ncols=4, figsize=(18, 4.4))
plot_field(
    axes[0],
    rmse_field,
    title="RMSE",
    cmap="magma_r",
    points=POINTS,
    cbar_label="mm/day",
)
for ax, field, name in zip(
    axes[1:],
    [bias_mean_field, bias_p98_field, bias_rx1_field],
    ["mean", "P98", "RX1day"],
):
    v = float(np.nanmax(np.abs(field))) * 0.8
    plot_field(
        ax,
        field,
        title=f"Bias of {name} (model - truth)",
        cmap="RdBu",
        vmin=-v,
        vmax=v,
        points=POINTS,
        cbar_label="mm/day",
    )
plt.show()
