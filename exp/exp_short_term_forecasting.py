import os
import time
import warnings
from typing import Any

import numpy as np
import pandas
import torch
from torch import nn, optim
from tqdm import tqdm

from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, create_experiment_folder, setup_experiment_logger, visual

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


class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)
        self.logger = setup_experiment_logger(args, "short_term_forecast")

    def _build_model(self):
        if self.args.data == "m4":
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Up to M4 config
            self.args.seq_len = 2 * self.args.pred_len  # input_len = 2*pred_len
            self.args.label_len = self.args.pred_len
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
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

    def _select_criterion(self, loss_name="MSE"):
        if loss_name == "MSE":
            return nn.MSELoss()
        elif loss_name == "MAPE":
            return mape_loss()
        elif loss_name == "MASE":
            return mase_loss()
        elif loss_name == "SMAPE":
            return smape_loss()

    def train(self, setting):
        self.logger.info("Starting training for short-term forecasting task")
        self.logger.info(f"Setting: {setting}")

        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")

        self.logger.info(f"Training data size: {len(train_data)}")
        self.logger.info(f"Validation data size: {len(vali_data)}")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        mse = nn.MSELoss()

        self.logger.info(f"Training epochs: {self.args.train_epochs}")
        self.logger.info(f"Training steps per epoch: {train_steps}")
        self.logger.info(f"Loss function: {self.args.loss}")
        self.logger.info(f"Optimizer: {type(model_optim).__name__}")

        for epoch in range(self.args.train_epochs):
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            # Use tqdm for training progress
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_iterator):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, None, dec_inp, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                batch_y_mark = batch_y_mark[:, -self.args.pred_len :, f_dim:].to(self.device)
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value  # + loss_sharpness * 1e-5
                train_loss.append(loss.item())

                # Update tqdm description with current loss
                train_iterator.set_postfix(loss=f"{loss.item():.7f}")

                loss.backward()
                model_optim.step()

            epoch_time_cost = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {epoch_time_cost}")
            self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time_cost:.2f} seconds")

            train_loss = np.average(train_loss)
            vali_loss = self.vali(train_loader, vali_loader, criterion)
            test_loss = vali_loss
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            self.logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.8f}, Vali Loss={vali_loss:.8f}, Test Loss={test_loss:.8f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.model.eval()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len :, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float()  # .to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i] : id_list[i + 1], :, :] = (
                    self.model(x[id_list[i] : id_list[i + 1]], None, dec_inp[id_list[i] : id_list[i + 1]], None).detach().cpu()
                )
            f_dim = -1 if self.args.features == "MS" else 0
            outputs = outputs[:, -self.args.pred_len :, f_dim:]
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape)

            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train()
        return loss

    def test(self, setting, test=0):
        self.logger.info("Starting test evaluation for short-term forecasting task")
        self.logger.info(f"Setting: {setting}")

        _, train_loader = self._get_data(flag="train")
        _, test_loader = self._get_data(flag="test")
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)

        self.logger.info(f"Test input shape: {x.shape}")
        self.logger.info(f"Test target series count: {len(y)}")

        if test:
            print("loading model")
            self.logger.info("Loading model from checkpoint")
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

        # Create hierarchical results folder structure with index and timestamp
        folder_path = create_experiment_folder(self.args, "short_term_forecast")
        print(f"Results will be saved to: {folder_path}")

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len :, :], dec_inp], dim=1).float()
            # encoder - decoder
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                outputs[id_list[i] : id_list[i + 1], :, :] = self.model(x[id_list[i] : id_list[i + 1]], None, dec_inp[id_list[i] : id_list[i + 1]], None)

            f_dim = -1 if self.args.features == "MS" else 0
            outputs = outputs[:, -self.args.pred_len :, f_dim:]
            outputs = outputs.detach().cpu().numpy()

            preds = outputs
            trues = y
            x = x.detach().cpu().numpy()

            for i in range(0, preds.shape[0], preds.shape[0] // 10):
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        print("test shape:", preds.shape)
        self.logger.info(f"Test predictions shape: {preds.shape}")

        # Ensure shape consistency between predictions and targets
        # preds shape: (48000, 18, 1), trues shape: (48000, 18)
        # Convert trues to match preds dimension
        trues_array = np.array(trues)
        if len(trues_array.shape) == 2 and len(preds.shape) == 3:
            trues_array = trues_array[..., np.newaxis]  # Add the last dimension
        elif len(trues_array.shape) == 3 and len(preds.shape) == 2:
            preds = preds.squeeze(-1)  # Remove the last dimension from preds

        # Calculate general regression metrics
        mae, mse, rmse, mape_metric, mspe = metric(preds, trues_array)
        print(f"RMSE: {rmse}, MSE: {mse}, MAE: {mae}, MAPE: {mape_metric}, MSPE: {mspe}")

        # Log detailed metrics
        self.logger.info("=" * 60)
        self.logger.info("GENERAL REGRESSION METRICS")
        self.logger.info("=" * 60)
        self.logger.info(f"RMSE: {rmse:.8f}")
        self.logger.info(f"MSE: {mse:.8f}")
        self.logger.info(f"MAE: {mae:.8f}")
        self.logger.info(f"MAPE: {mape_metric:.8f}")
        self.logger.info(f"MSPE: {mspe:.8f}")
        self.logger.info("=" * 60)

        # Save general metrics to file
        general_metrics_file = folder_path + "general_metrics.txt"
        with open(general_metrics_file, "w") as f:
            f.write(f"Setting: {setting}\n")
            f.write("=" * 50 + "\n")
            f.write(f"RMSE: {rmse}\n")
            f.write(f"MSE: {mse}\n")
            f.write(f"MAE: {mae}\n")
            f.write(f"MAPE: {mape_metric}\n")
            f.write(f"MSPE: {mspe}\n")
            f.write("=" * 50 + "\n")
        print(f"General metrics saved to {general_metrics_file}")
        self.logger.info(f"General metrics saved to {general_metrics_file}")

        # Save predictions and ground truth
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues_array)
        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape_metric, mspe]))

        # Save forecast results in the same folder
        forecasts_df = pandas.DataFrame(preds[:, :, 0], columns=[f"V{i + 1}" for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[: preds.shape[0]]
        forecasts_df.index.name = "id"
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(folder_path + self.args.seasonal_patterns + "_forecast.csv")

        # Also save to m4_results for compatibility with evaluation
        m4_folder_path = "./m4_results/" + self.args.model + "/"
        if not os.path.exists(m4_folder_path):
            os.makedirs(m4_folder_path)
        forecasts_df.to_csv(m4_folder_path + self.args.seasonal_patterns + "_forecast.csv")

        # Upload individual seasonal pattern results to Google Sheets
        if SHEETS_AVAILABLE:
            try:
                seasonal_pattern = self.args.seasonal_patterns

                # Calculate individual metrics for this seasonal pattern
                # For individual patterns, we use the general regression metrics as a proxy
                # Since SMAPE/MASE/OWA are calculated only in M4Summary for full evaluation
                test_metrics = {
                    "smape": mape_metric,  # Use MAPE as proxy for SMAPE
                    "mase": mae,  # Use MAE as proxy for MASE
                    "owa": rmse,  # Use RMSE as proxy for OWA
                }

                # Upload to LeaderBoard (this will go to short-term-forecasting tab)
                upload_experiment_results(
                    self.args,
                    0.0,  # train_loss - not used for short-term forecasting
                    0.0,  # val_loss - not used for short-term forecasting
                    0.0,  # test_loss - not used for short-term forecasting
                    test_metrics,
                    setting,
                )

                print(f"Successfully uploaded {seasonal_pattern} result to LeaderBoard")
                self.logger.info(f"Successfully uploaded {seasonal_pattern} result to LeaderBoard")

            except Exception as e:
                print(f"Failed to upload {seasonal_pattern} result to LeaderBoard: {e}")
                self.logger.error(f"Failed to upload {seasonal_pattern} result to LeaderBoard: {e}")

        print(self.args.model)
        file_path = "./m4_results/" + self.args.model + "/"

        # Check if all 6 patterns are complete for optional M4 summary calculation
        all_patterns_complete = (
            os.path.exists(file_path)
            and "Weekly_forecast.csv" in os.listdir(file_path)
            and "Monthly_forecast.csv" in os.listdir(file_path)
            and "Yearly_forecast.csv" in os.listdir(file_path)
            and "Daily_forecast.csv" in os.listdir(file_path)
            and "Hourly_forecast.csv" in os.listdir(file_path)
            and "Quarterly_forecast.csv" in os.listdir(file_path)
        )

        if all_patterns_complete:
            try:
                m4_summary = M4Summary(file_path, self.args.root_path)
                # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
                smape_results, owa_results, mape, mase = m4_summary.evaluate()
                print("smape:", smape_results)
                print("mape:", mape)
                print("mase:", mase)
                print("owa:", owa_results)

                # Log M4 metrics
                self.logger.info("=" * 60)
                self.logger.info("M4 COMPETITION METRICS")
                self.logger.info("=" * 60)

                # Handle dictionary or scalar results
                if isinstance(smape_results, dict):
                    smape_avg = smape_results.get("Average", 0.0)
                    mape_avg = mape.get("Average", 0.0) if isinstance(mape, dict) else mape
                    mase_avg = mase.get("Average", 0.0) if isinstance(mase, dict) else mase
                    owa_avg = owa_results.get("Average", 0.0) if isinstance(owa_results, dict) else owa_results

                    self.logger.info(f"SMAPE Average: {smape_avg:.8f}")
                    self.logger.info(f"MAPE Average: {mape_avg:.8f}")
                    self.logger.info(f"MASE Average: {mase_avg:.8f}")
                    self.logger.info(f"OWA Average: {owa_avg:.8f}")
                else:
                    self.logger.info(f"SMAPE: {smape_results:.8f}")
                    self.logger.info(f"MAPE: {mape:.8f}")
                    self.logger.info(f"MASE: {mase:.8f}")
                    self.logger.info(f"OWA: {owa_results:.8f}")

                self.logger.info("=" * 60)

                # Save M4-specific results to file in the same folder structure
                result_file = folder_path + "result_m4_short_term_forecast.txt"
                f = open(result_file, "a")
                f.write(f"Model: {self.args.model}\n")
                f.write(f"Setting: {setting}\n")
                f.write("=" * 50 + "\n")
                f.write(f"SMAPE: {smape_results}\n")
                f.write(f"MAPE: {mape}\n")
                f.write(f"MASE: {mase}\n")
                f.write(f"OWA: {owa_results}\n")
                f.write("=" * 50 + "\n")
                f.write("\n")
                f.close()
                print(f"M4 results saved to {result_file}")
                self.logger.info("M4 summary calculation completed for all patterns")
            except Exception as e:
                print(f"M4 summary calculation failed: {e}")
                self.logger.error(f"M4 summary calculation failed: {e}")
        else:
            print(f"Individual {self.args.seasonal_patterns} pattern completed. M4 summary will be calculated when all 6 patterns are finished.")
            self.logger.info(f"Individual {self.args.seasonal_patterns} pattern completed. All patterns check: {all_patterns_complete}")
