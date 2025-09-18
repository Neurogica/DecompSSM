import numpy as np
from tqdm import tqdm


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    # For memory efficiency, avoid creating large intermediate arrays
    if pred.size > 1e7:  # If array is very large (>10M elements)
        # Process in chunks to avoid memory issues
        chunk_size = 10000000  # 10M elements per chunk (increased for better performance)
        mae_sum = 0.0
        total_elements = 0

        flat_pred = pred.flatten()
        flat_true = true.flatten()

        for i in range(0, len(flat_pred), chunk_size):
            end_idx = min(i + chunk_size, len(flat_pred))
            chunk_pred = flat_pred[i:end_idx]
            chunk_true = flat_true[i:end_idx]

            mae_sum += np.sum(np.abs(chunk_pred - chunk_true))
            total_elements += len(chunk_pred)

        return mae_sum / total_elements
    else:
        return np.mean(np.abs(pred - true))


def MSE(pred, true):
    # For memory efficiency, avoid creating large intermediate arrays
    if pred.size > 1e7:  # If array is very large (>10M elements)
        # Process in chunks to avoid memory issues
        chunk_size = 10000000  # 10M elements per chunk (increased for better performance)
        mse_sum = 0.0
        total_elements = 0

        flat_pred = pred.flatten()
        flat_true = true.flatten()

        for i in range(0, len(flat_pred), chunk_size):
            end_idx = min(i + chunk_size, len(flat_pred))
            chunk_pred = flat_pred[i:end_idx]
            chunk_true = flat_true[i:end_idx]

            diff = chunk_pred - chunk_true
            mse_sum += np.sum(diff * diff)
            total_elements += len(chunk_pred)

        return mse_sum / total_elements
    else:
        return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    # Avoid division by zero and handle inf/nan values
    mask = np.abs(true) > 1e-10  # Only consider non-zero true values
    if np.sum(mask) == 0:
        return 0.0
    mape_values = np.abs((pred[mask] - true[mask]) / true[mask])
    # Filter out inf and nan values
    mape_values = mape_values[np.isfinite(mape_values)]
    return np.mean(mape_values) if len(mape_values) > 0 else 0.0


def MSPE(pred, true):
    # Avoid division by zero and handle inf/nan values
    mask = np.abs(true) > 1e-10  # Only consider non-zero true values
    if np.sum(mask) == 0:
        return 0.0
    mspe_values = np.square((pred[mask] - true[mask]) / true[mask])
    # Filter out inf and nan values
    mspe_values = mspe_values[np.isfinite(mspe_values)]
    return np.mean(mspe_values) if len(mspe_values) > 0 else 0.0


def metric(pred, true):
    print(f"Starting metric calculation for arrays with shape: pred={pred.shape}, true={true.shape}")

    # Calculate memory usage
    memory_per_array = pred.nbytes / (1024**3)  # GB
    print(f"Memory per array: {memory_per_array:.2f} GB")

    # Use batch processing for large arrays to avoid memory issues
    batch_size = min(100, pred.shape[0])  # Process in batches of 100 samples max
    n_batches = (pred.shape[0] + batch_size - 1) // batch_size

    print(f"Processing in {n_batches} batches of size {batch_size}")

    mae_values = []
    mse_values = []
    mape_values = []
    mspe_values = []

    # Use tqdm to show progress
    for i in tqdm(range(n_batches), desc="Calculating metrics", unit="batch"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, pred.shape[0])

        pred_batch = pred[start_idx:end_idx]
        true_batch = true[start_idx:end_idx]

        # Calculate metrics for this batch
        mae_batch = MAE(pred_batch, true_batch)
        mse_batch = MSE(pred_batch, true_batch)
        mape_batch = MAPE(pred_batch, true_batch)
        mspe_batch = MSPE(pred_batch, true_batch)

        mae_values.append(mae_batch)
        mse_values.append(mse_batch)
        mape_values.append(mape_batch)
        mspe_values.append(mspe_batch)

    # Calculate final metrics as weighted averages
    print("Calculating final metrics...")
    mae = np.mean(mae_values)
    mse = np.mean(mse_values)
    rmse = np.sqrt(mse)
    mape = np.mean(mape_values)
    mspe = np.mean(mspe_values)

    print(f"Final metrics - MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.6f}, MSPE: {mspe:.6f}")
    print("Metric calculation completed")
    return mae, mse, rmse, mape, mspe
