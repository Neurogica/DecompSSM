import torch.multiprocessing
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment, create_experiment_folder, setup_experiment_logger

torch.multiprocessing.set_sharing_strategy("file_system")
import os
import time
import warnings

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

warnings.filterwarnings("ignore")


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.logger = setup_experiment_logger(args, "anomaly_detection")

    def _build_model(self):
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
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}. Options: [adam, adamw, sgd]")

        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        self.logger.info("Starting training for anomaly detection task")
        self.logger.info(f"Setting: {setting}")

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

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
        self.logger.info(f"Optimizer: {type(model_optim).__name__}")
        self.logger.info(f"Criterion: {type(criterion).__name__}")

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # Use tqdm for training progress
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")
            for i, (batch_x, batch_y) in enumerate(train_iterator):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                # Update tqdm description with current loss
                train_iterator.set_postfix(loss=f"{loss.item():.7f}")

                loss.backward()
                model_optim.step()

            epoch_time_cost = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {epoch_time_cost}")
            self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time_cost:.2f} seconds")

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            self.logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.8f}, Vali Loss={vali_loss:.8f}, Test Loss={test_loss:.8f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                self.logger.info("Early stopping triggered")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        self.logger.info("Starting test evaluation for anomaly detection task")
        self.logger.info(f"Setting: {setting}")

        test_data, test_loader = self._get_data(flag="test")
        train_data, train_loader = self._get_data(flag="train")
        if test:
            print("loading model")
            self.logger.info("Loading model from checkpoint")
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

        self.logger.info(f"Test data size: {len(test_data)}")
        self.logger.info(f"Train data size: {len(train_data)}")
        self.logger.info(f"Test batches: {len(test_loader)}")

        attens_energy = []
        # Create hierarchical results folder structure with index and timestamp
        folder_path = create_experiment_folder(self.args, "anomaly_detection")
        print(f"Results will be saved to: {folder_path}")

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average="binary")
        print(f"Accuracy : {accuracy:0.4f}, Precision : {precision:0.4f}, Recall : {recall:0.4f}, F-score : {f_score:0.4f} ")

        # Save results to file in the same folder structure
        result_file = folder_path + "result_anomaly_detection.txt"
        f = open(result_file, "a")
        f.write(f"Setting: {setting}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy : {accuracy:0.4f}, Precision : {precision:0.4f}, Recall : {recall:0.4f}, F-score : {f_score:0.4f}\n")
        f.write("=" * 50 + "\n")
        f.write("\n")
        f.close()
        print(f"Results saved to {result_file}")
