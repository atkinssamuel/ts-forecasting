import torch.nn as nn
import torch
import time
import os
import json

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
):
    out = {}
    model.eval()

    train_batcher = iter(train_loader)
    valid_batcher = iter(valid_loader)

    for split, batcher in zip(["train", "valid"], [train_loader, valid_loader]):
        n_iters = min(100, len(batcher))

        losses = torch.zeros(n_iters)

        for k in range(n_iters):
            xb, yb = next(train_batcher if split == "train" else valid_batcher)

            _, loss = model(xb, yb)
            losses[k] = loss.mean()

        out[split] = losses.mean()

    model.train()
    return out


def is_designated(device):
    return (
        device == "cpu"
        or device == 0
        or (isinstance(device, torch.device) and device.index == 0)
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    hyperparameters: dict,
    valid_loader: Union[DataLoader, None] = None,
    device: Union[int, str] = "cpu",
    model_dir="models/",
    verbose: bool = True,
):
    print()

    # extracting hyperparameters

    print("Extracting hyperparameters...")

    for k in ["n_updates", "checkpoint_iter"]:
        assert k in hyperparameters

    n_updates = hyperparameters["n_updates"]
    checkpoint_iter = hyperparameters["checkpoint_iter"]

    # setting vars

    min_valid_loss = float("infinity")
    device_str = "GPU-" + str(device) if device != "cpu" else "CPU"

    # initializing writer and time constants

    print("Initializing writer...")

    writer = None

    if is_designated(device):
        start_time = time.time()

        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        writer = SummaryWriter(model_dir)

        with open(model_dir + "hyperparameters.json", "w") as f:
            f.write(json.dumps(hyperparameters))

    # moving model to the appropriate device (if it has not been done already)

    print("Moving model to DDP if distributed...")

    model = model.to(device)
    if device != "cpu":
        model = DDP(model, device_ids=[device])

    # initializing loop arrays

    print("Creating iterators...")

    train_batcher = iter(train_loader)
    if valid_loader:
        valid_batcher = iter(valid_loader)

    print()

    print("Training setup complete...")

    for i in range(n_updates):
        try:
            xb, yb = next(train_batcher)
        except:
            train_batcher = iter(train_loader)
            xb, yb = next(train_batcher)

        # --- ADD THESE DEBUG PRINTS ---
        print(f"\n--- Debugging Iteration {i} ---")
        print(f"xb shape: {xb.shape}, dtype: {xb.dtype}")
        print(f"yb shape: {yb.shape}, dtype: {yb.dtype}")
        print(f"xb min: {xb.min():.4f}, max: {xb.max():.4f}, mean: {xb.mean():.4f}")
        print(f"yb min: {yb.min():.4f}, max: {yb.max():.4f}, mean: {yb.mean():.4f}")
        # If your model is LSTMModel or CNN1DModel:
        # Check the CUSIP IDs specifically
        print(
            f"Ticker IDs (xb[:, 0]) min: {xb[:, 0].min().item()}, max: {xb[:, 0].max().item()}"
        )
        # Check historical data part
        print(
            f"Historical VWAP (xb[:, 1:]) min: {xb[:, 1:].min():.4f}, max: {xb[:, 1:].max():.4f}"
        )
        # --- END DEBUG PRINTS ---

        try:
            predictions, loss = model(xb, yb)  # Changed _ to predictions for inspection

            # --- ADD THESE DEBUG PRINTS AFTER MODEL FORWARD ---
            print(f"Predictions shape: {predictions.shape}, dtype: {predictions.dtype}")
            print(
                f"Predictions min: {predictions.min():.4f}, max: {predictions.max():.4f}, mean: {predictions.mean():.4f}"
            )
            print(f"Loss value: {loss.item():.4f}")
            # --- END DEBUG PRINTS ---

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            optimizer.step()
        except Exception as e:
            print()
            print(f"Exception during training: {e}")
            print(f"X: {xb.shape}, y: {yb.shape}")
            print()

            continue

        # logging with designated device

        if is_designated(device):
            train_loss = loss.mean().item()
            total_time = time.time() - start_time

            writer.add_scalar("Loss/train", train_loss, i)
            writer.add_scalar("Constants/time", total_time, i)

            if valid_loader:
                model.eval()

                with torch.no_grad():
                    try:
                        xbv, ybv = next(valid_batcher)
                    except:
                        valid_batcher = iter(valid_loader)
                        xbv, ybv = next(valid_batcher)

                _, loss = model(xbv, ybv)
                valid_loss = loss.mean().item()

                model.train()

                writer.add_scalar("Loss/valid", valid_loss, i)

            if verbose:
                print(
                    f"{device_str} - step {i}: "
                    f"train loss = {train_loss:.4f},{f' valid loss = {valid_loss:.4f},'if valid_loader else ''}"
                    f"total time = {total_time:.1f}s"
                )

        # checkpointing with designated device

        if i % checkpoint_iter == 0 and is_designated(device):
            print(f"Checkpointing at iteration {i}")

            torch.save(
                model.state_dict() if device == "cpu" else model.module.state_dict(),
                model_dir + "latest.model",
            )

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss

                print(f"New optimal model found at iteration {i}")

                torch.save(
                    (
                        model.state_dict()
                        if device == "cpu"
                        else model.module.state_dict()
                    ),
                    model_dir + "optimal.model",
                )
