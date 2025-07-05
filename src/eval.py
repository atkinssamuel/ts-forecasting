import numpy as np
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def eval_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode (disables dropout, batch norm updates)

    all_predictions = []
    all_true_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions, _ = model(X_batch, y_batch)

            ticker_ind = X_batch[0]

            all_predictions.append(predictions.cpu().numpy())
            all_true_targets.append(y_batch.cpu().numpy())

    predictions_np = np.concatenate(all_predictions, axis=0).flatten()
    true_targets_np = np.concatenate(all_true_targets, axis=0).flatten()

    mse = mean_squared_error(true_targets_np, predictions_np)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_targets_np, predictions_np)
    r2 = r2_score(true_targets_np, predictions_np)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": predictions_np,
        "true_targets": true_targets_np,
    }
