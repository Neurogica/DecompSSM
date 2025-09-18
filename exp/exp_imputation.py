import os
import time
import warnings

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, create_experiment_folder, setup_experiment_logger, visual

warnings.filterwarnings("ignore")


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)
        # Setup logger for imputation task
        self.logger = setup_experiment_logger(args, "imputation")

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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader, desc="Validation")):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                """
                B = batch size
                T = seq len
                N = number of features
                """
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                mask = mask.detach().cpu()

                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        self.logger.debug(f"Validation completed: Avg Loss={total_loss:.8f}, Batches={len(vali_loader)}")
        return total_loss

    def train(self, setting):
        self.logger.info("Starting training for imputation task")
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
        self.logger.info(f"Mask rate: {self.args.mask_rate}")
        self.logger.info(f"Optimizer: {type(model_optim).__name__}")
        self.logger.info(f"Criterion: {type(criterion).__name__}")

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # Use tqdm for training progress
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_iterator):
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                outputs = self.model(inp, batch_x_mark, None, None, mask)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
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
        self.logger.info("Starting test evaluation for imputation task")
        self.logger.info(f"Setting: {setting}")

        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.logger.info("Loading model from checkpoint")
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

        self.logger.info(f"Test data size: {len(test_data)}")
        self.logger.info(f"Test batches: {len(test_loader)}")

        preds = []
        trues = []
        masks = []

        # Create hierarchical results folder structure with index and timestamp
        folder_path = create_experiment_folder(self.args, "imputation")
        print(f"Results will be saved to: {folder_path}")
        self.logger.info(f"Results will be saved to: {folder_path}")

        self.model.eval()
        with torch.no_grad():
            # Use tqdm for test progress
            test_iterator = tqdm(test_loader, desc="Testing")
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_iterator):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # random mask
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x.masked_fill(mask == 0, 0)

                # imputation
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # eval
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, :, f_dim:]

                # add support for MS
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                outputs = outputs.detach().cpu().numpy()
                pred = outputs
                true = batch_x.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                masks.append(mask.detach().cpu())

                if i % 20 == 0:
                    filled = true[0, :, -1].copy()
                    filled = filled * mask[0, :, -1].detach().cpu().numpy() + pred[0, :, -1] * (1 - mask[0, :, -1].detach().cpu().numpy())
                    visual(true[0, :, -1], filled, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print("test shape:", preds.shape, trues.shape)
        self.logger.info(f"Test prediction shape: {preds.shape}")
        self.logger.info(f"Test ground truth shape: {trues.shape}")

        # result save
        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print(f"mse:{mse}, mae:{mae}")
        self.logger.info(f"Test Results - MSE: {mse:.8f}, MAE: {mae:.8f}, RMSE: {rmse:.8f}, MAPE: {mape:.8f}, MSPE: {mspe:.8f}")

        # Save results to the hierarchical folder structure
        result_file = os.path.join(folder_path, "result_imputation.txt")
        with open(result_file, "w") as f:
            f.write(f"Setting: {setting}\n")
            f.write(f"MSE: {mse:.8f}\n")
            f.write(f"MAE: {mae:.8f}\n")
            f.write(f"RMSE: {rmse:.8f}\n")
            f.write(f"MAPE: {mape:.8f}\n")
            f.write(f"MSPE: {mspe:.8f}\n")

        np.save(os.path.join(folder_path, "metrics.npy"), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, "pred.npy"), preds)
        np.save(os.path.join(folder_path, "true.npy"), trues)
        np.save(os.path.join(folder_path, "masks.npy"), masks)

        self.logger.info(f"All results saved to: {folder_path}")
