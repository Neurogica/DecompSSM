import logging
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                elif pred[j] == 0:
                    pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                elif pred[j] == 0:
                    pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def get_next_experiment_index(base_path):
    """
    Get the next experiment index by counting existing folders.
    """

    if not os.path.exists(base_path):
        return 1

    existing_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.split("_")[0].isdigit()]

    if not existing_folders:
        return 1

    # Extract indices and find the maximum
    indices = []
    for folder in existing_folders:
        try:
            index = int(folder.split("_")[0])
            indices.append(index)
        except (ValueError, IndexError):
            continue

    return max(indices) + 1 if indices else 1


def create_experiment_folder(args, task_name=None):
    """
    Create a hierarchical experiment folder with index and timestamp.

    Format: results/{task_name}/{dataset}/{model}/{dimensions}/{index_timestamp}/
    """
    import os

    # Use task_name from args if not provided
    if task_name is None:
        task_name = getattr(args, "task_name", "experiment")

    # Determine dimensions based on task and dataset
    if args.data == "m4":
        dimensions = getattr(args, "seasonal_patterns", "Unknown")
    elif hasattr(args, "seq_len") and hasattr(args, "pred_len"):
        if task_name == "classification":
            dimensions = str(args.seq_len)
        else:
            dimensions = f"{args.seq_len}_{args.pred_len}"
    else:
        dimensions = "default"

    # Create base path
    base_path = f"./results/{task_name}/{args.data}/{args.model}/{dimensions}"

    # Get next index
    next_index = get_next_experiment_index(base_path)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create final folder path
    folder_name = f"{next_index}_{timestamp}"
    folder_path = os.path.join(base_path, folder_name)

    # Create directory
    os.makedirs(folder_path, exist_ok=True)

    return folder_path + "/"


def setup_experiment_logger(args, task_name=None, logger_name="experiment"):
    """
    Set up logger for experiment with hierarchical log file structure.
    Pattern: logs/{task_name}/{dataset}/{model}/{input_len_output_len}/{trial_name}_{timestamp}.log
    """

    if task_name is None:
        task_name = getattr(args, "task_name", "experiment")

    # Create dimensions string for log file naming
    if args.data == "m4":
        dimensions = getattr(args, "seasonal_patterns", "Unknown")
    elif hasattr(args, "seq_len") and hasattr(args, "pred_len"):
        if task_name == "classification":
            dimensions = str(args.seq_len)
        else:
            dimensions = f"{args.seq_len}_{args.pred_len}"
    else:
        dimensions = "default"

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get trial name with date
    trial_name = getattr(args, "trial_name", "")
    if trial_name:
        # Add date to trial name if not already present
        date_str = datetime.now().strftime("%Y%m%d")
        if date_str not in trial_name:
            trial_name = f"{trial_name}_{date_str}"
        log_filename = f"{trial_name}_{timestamp}.log"
    else:
        # Fallback to timestamp only if no trial name
        log_filename = f"{timestamp}.log"

    # Log directory structure
    log_dir = os.path.join("logs", task_name, args.data, args.model, dimensions)
    os.makedirs(log_dir, exist_ok=True)

    # Log file path
    log_file = os.path.join(log_dir, log_filename)

    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Experiment logger initialized: {log_file}")
    logger.info(f"Task: {task_name}, Dataset: {args.data}, Model: {args.model}")
    if hasattr(args, "trial_name") and args.trial_name:
        logger.info(f"Trial Name: {args.trial_name}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("=" * 80)

    return logger
