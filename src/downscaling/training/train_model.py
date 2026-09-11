import torch
import torch.nn as nn


def evaluate_loss(model, loader, loss_fn):
    model.eval()
    total, n = 0.0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            total += loss_fn(model(xb), yb).item() * xb.size(0)
            n += xb.size(0)

    return total / n


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=NUM_EPOCHS,
    lr=LEARNING_RATE,
    verbose=True,
):
    """Train `model` with Adam + MSE, returning the train/val loss history.

    This function is model-agnostic: pass any nn.Module mapping (N, C, H, W) -> (N, n_out)
    and it will train it. Re-use it for your own architectures.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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
                f"train MSE {history['train'][-1]:.3f}  val MSE {history['val'][-1]:.3f}"
            )

    return history


model = build_model()

print(f"Training DeepESD from scratch for {NUM_EPOCHS} epochs on {DEVICE} ...")

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

fig, ax = plt.subplots(figsize=(6, 4))
epochs = range(1, len(history["train"]) + 1)
ax.plot(epochs, history["train"], label="training")
ax.plot(epochs, history["val"], label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE loss (mm/day)$^2$")
ax.set_title("Training and validation loss")
ax.legend()
plt.show()


def downscale(model, predictors_ds, mean=None, std=None, batch=512):
    """Apply a trained model to a predictor Dataset -> precipitation DataArray (time, lat, lon).

    Predictors are standardized with the TRAINING statistics (mean/std), which is what we
    must do operationally: at projection time we only know the training-period statistics.
    """
    mean = PRED_MEAN if mean is None else mean
    std = PRED_STD if std is None else std
    x = predictors_to_array(predictors_ds, mean, std)

    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.as_tensor(x[i : i + batch]).to(DEVICE)
            chunks.append(model(xb).cpu().numpy())
    pred = np.concatenate(chunks, axis=0)
    pred = pred.reshape(pred.shape[0], TARGET_LAT.size, TARGET_LON.size)

    da = xr.DataArray(
        pred,
        dims=("time", "lat", "lon"),
        coords={"time": predictors_ds["time"], "lat": TARGET_LAT, "lon": TARGET_LON},
        name="pr",
    )

    # Precipitation cannot be negative and an MSE-trained model can produce small negatives
    da = da.clip(min=0.0)
    da.attrs["units"] = "mm day-1"
    return da


def align_time(pred, obs):
    """Make sure prediction and reference share the same time axis (same period)."""
    assert pred.sizes["time"] == obs.sizes["time"], (
        "prediction/reference length mismatch"
    )
    return pred.assign_coords(time=obs["time"].values)
