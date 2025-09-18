import os
import time
import warnings
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy, create_experiment_folder, setup_experiment_logger

try:
    from utils.sheets_uploader import upload_experiment_results

    SHEETS_AVAILABLE = True
except ImportError:
    SHEETS_AVAILABLE = False
    print("Warning: Google Sheets uploader not available")

    def upload_experiment_results(args: Any, train_loss: float, val_loss: float, test_loss: float, test_metrics: dict[str, float], setting: str) -> None:
        """Dummy function when sheets are not available"""
        pass


warnings.filterwarnings("ignore")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.logger = setup_experiment_logger(args, "classification")
        # Dictionary to store results for multiple datasets in one trial
        self.dataset_results = {}

    def _build_model(self):
        # model input depends on data
        train_data, train_loader = self._get_data(flag="TRAIN")
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        # model init
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer_type = getattr(self.args, "optimizer", "adam").lower()

        if optimizer_type == "adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif optimizer_type == "adamw":
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)
        elif optimizer_type == "sgd":
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1e-4)
        elif optimizer_type == "radam":
            model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}. Options: [adam, adamw, sgd, radam]")

        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.long().squeeze().cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        self.logger.info("Starting training for classification task")
        self.logger.info(f"Setting: {setting}")

        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="TEST")
        test_data, test_loader = self._get_data(flag="TEST")

        self.logger.info(f"Training data size: {len(train_data)}")
        self.logger.info(f"Validation data size: {len(vali_data)}")
        self.logger.info(f"Test data size: {len(test_data)}")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        self.logger.info(f"Training epochs: {self.args.train_epochs}")
        self.logger.info(f"Training steps per epoch: {train_steps}")

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # Use tqdm for training progress
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")
            for i, (batch_x, label, padding_mask) in enumerate(train_iterator):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_loss.append(loss.item())

                # Update tqdm description with current loss
                train_iterator.set_postfix(loss=f"{loss.item():.7f}")

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            epoch_time_cost = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {epoch_time_cost}")
            self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time_cost:.2f} seconds")

            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.3f} Vali Loss: {vali_loss:.3f} Vali Acc: {val_accuracy:.3f} Test Loss: {test_loss:.3f} Test Acc: {test_accuracy:.3f}"
            )
            self.logger.info(
                f"Epoch {epoch + 1}: Train Loss={train_loss:.6f}, Vali Loss={vali_loss:.6f}, Vali Acc={val_accuracy:.6f}, Test Loss={test_loss:.6f}, Test Acc={test_accuracy:.6f}"
            )
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                self.logger.info("Early stopping triggered")
                break
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        self.logger.info("Starting test evaluation for classification task")
        self.logger.info(f"Setting: {setting}")

        test_data, test_loader = self._get_data(flag="TEST")
        if test:
            print("loading model")
            self.logger.info("Loading model from checkpoint")
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

        self.logger.info(f"Test data size: {len(test_data)}")
        self.logger.info(f"Test batches: {len(test_loader)}")

        preds = []
        trues = []
        # Create hierarchical results folder structure with index and timestamp
        folder_path = create_experiment_folder(self.args, "classification")
        print(f"Results will be saved to: {folder_path}")
        self.logger.info(f"Results will be saved to: {folder_path}")

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print("test shape:", preds.shape, trues.shape)
        self.logger.info(f"Test predictions shape: {preds.shape}, true labels shape: {trues.shape}")

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # Result save - use the same folder path that was created earlier

        print(f"accuracy:{accuracy}")
        self.logger.info(f"Final Test Accuracy: {accuracy:.6f}")

        # Store result for this dataset and upload to LeaderBoard
        dataset_name = getattr(self.args, "model_id", getattr(self.args, "data", "Unknown"))

        # Upload to LeaderBoard with classification-specific metrics
        if SHEETS_AVAILABLE:
            try:
                # Create test metrics with only accuracy
                test_metrics = {"accuracy": accuracy}

                # Upload to LeaderBoard (this will go to classification tab)
                upload_experiment_results(
                    self.args,
                    0.0,  # train_loss - not used for classification
                    0.0,  # val_loss - not used for classification
                    0.0,  # test_loss - not used for classification
                    test_metrics,
                    setting,
                )

                print(f"Successfully uploaded {dataset_name} result to LeaderBoard")
                self.logger.info(f"Successfully uploaded {dataset_name} result to LeaderBoard")

            except Exception as e:
                print(f"Failed to upload {dataset_name} result to LeaderBoard: {e}")
                self.logger.error(f"Failed to upload {dataset_name} result to LeaderBoard: {e}")

        # Store result locally as well
        self.dataset_results[dataset_name] = accuracy
        print(f"Stored result for {dataset_name}: {accuracy:.6f}")
        self.logger.info(f"Stored result for {dataset_name}: {accuracy:.6f}")

        # Save results to file in the same folder structure
        result_file = folder_path + "result_classification.txt"
        f = open(result_file, "a")
        f.write(f"Setting: {setting}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write("=" * 50 + "\n")
        f.write("\n")
        f.close()
        print(f"Results saved to {result_file}")
        self.logger.info(f"Results saved to {result_file}")

        return accuracy
