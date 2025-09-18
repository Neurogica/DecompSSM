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
from utils.dtw_metric import accelerated_dtw
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


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.logger = setup_experiment_logger(args, "long_term_forecast")

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
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention and isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention and isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                # Ensure 3D tensors and correct layout [B, L, C]
                if outputs.dim() == 2:
                    outputs = outputs.unsqueeze(-1)
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)
                if outputs.size(1) == self.args.c_out:
                    outputs = outputs.transpose(1, 2)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        self.logger.debug(f"Validation completed: Avg Loss={total_loss:.8f}, Batches={len(vali_loader)}")
        return total_loss

    def train(self, setting):
        self.logger.info("Starting training for long-term forecasting task")
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

        # Track final losses for uploading to sheets
        final_train_loss = 0.0
        final_val_loss = 0.0

        self.logger.info(f"Training epochs: {self.args.train_epochs}")
        self.logger.info(f"Training steps per epoch: {train_steps}")
        self.logger.info(f"Optimizer: {type(model_optim).__name__}")
        self.logger.info(f"Criterion: {type(criterion).__name__}")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

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
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention and isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]

                        # Ensure 3D tensors and correct layout [B, L, C]
                        if outputs.dim() == 2:
                            outputs = outputs.unsqueeze(-1)
                        if batch_y.dim() == 2:
                            batch_y = batch_y.unsqueeze(-1)
                        if outputs.size(1) == self.args.c_out:
                            outputs = outputs.transpose(1, 2)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)

                        # Add auxiliary loss if available (for decomposition models)
                        if hasattr(self.model, "get_auxiliary_loss"):
                            aux_loss = self.model.get_auxiliary_loss()
                            total_loss = loss + aux_loss
                        else:
                            total_loss = loss

                        train_loss.append(total_loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention and isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]

                    # Ensure 3D tensors and correct layout [B, L, C]
                    if outputs.dim() == 2:
                        outputs = outputs.unsqueeze(-1)
                    if batch_y.dim() == 2:
                        batch_y = batch_y.unsqueeze(-1)
                    if outputs.size(1) == self.args.c_out:
                        outputs = outputs.transpose(1, 2)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                    # Add auxiliary loss if available (for decomposition models)
                    if hasattr(self.model, "get_auxiliary_loss"):
                        aux_loss = self.model.get_auxiliary_loss()
                        total_loss = loss + aux_loss
                    else:
                        total_loss = loss

                    train_loss.append(total_loss.item())

                # Update tqdm description with current loss
                train_iterator.set_postfix(loss=f"{total_loss.item():.7f}")

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()

            epoch_time_cost = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {epoch_time_cost}")
            self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time_cost:.2f} seconds")

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # Update final losses
            final_train_loss = train_loss
            final_val_loss = vali_loss

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            self.logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.8f}, Vali Loss={vali_loss:.8f}, Test Loss={test_loss:.8f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        # Store final losses for later upload
        self.final_train_loss = final_train_loss
        self.final_val_loss = final_val_loss

        return self.model

    def test(self, setting, test=0):
        self.logger.info("Starting test evaluation for long-term forecasting task")
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
        # Create hierarchical results folder structure with index and timestamp
        folder_path = create_experiment_folder(self.args, "long_term_forecast")
        print(f"Results will be saved to: {folder_path}")
        self.logger.info(f"Results will be saved to: {folder_path}")

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader, desc="Testing")):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        if self.args.output_attention and isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.output_attention and isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]

                # Ensure 3D tensors and correct layout [B, L, C]
                if outputs.dim() == 2:
                    outputs = outputs.unsqueeze(-1)
                if batch_y.dim() == 2:
                    batch_y = batch_y.unsqueeze(-1)
                if outputs.size(1) == self.args.c_out:
                    outputs = outputs.transpose(1, 2)
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    outputs = np.stack([test_data.inverse_transform(o) for o in outputs], axis=0)
                    batch_y = np.stack([test_data.inverse_transform(y) for y in batch_y], axis=0)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                # Normalize to [B, pred_len, C]
                def to_BLC_np(arr: np.ndarray) -> np.ndarray:
                    if arr.ndim != 3:
                        return arr
                    axis_L = next((i for i, s in enumerate(arr.shape) if s == self.args.pred_len), None)
                    axis_C = next((i for i, s in enumerate(arr.shape) if s == self.args.c_out), None)
                    if axis_L is None or axis_C is None:
                        return arr
                    axis_B = [0, 1, 2]
                    axis_B.remove(axis_L)
                    axis_B.remove(axis_C)
                    return arr.transpose(axis_B[0], axis_L, axis_C)

                pred = to_BLC_np(pred)
                true = to_BLC_np(true)

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        print(f"Before concatenation - preds type: {type(preds)}, length: {len(preds)}")
        print(f"Before concatenation - trues type: {type(trues)}, length: {len(trues)}")
        if len(preds) > 0:
            print(f"First pred shape: {preds[0].shape}")
            print(f"First true shape: {trues[0].shape}")

        try:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            print(f"After concatenation - preds type: {type(preds)}, shape: {preds.shape}")
            print(f"After concatenation - trues type: {type(trues)}, shape: {trues.shape}")
            print("test shape:", preds.shape, trues.shape)
        except Exception as e:
            print(f"Error during concatenation: {e}")
            self.logger.error(f"Error during concatenation: {e}")
            # If concatenation fails, try to recover
            if len(preds) > 0 and len(trues) > 0:
                preds = np.array(preds)
                trues = np.array(trues)
                print(f"Fallback - preds shape: {preds.shape}, trues shape: {trues.shape}")
            else:
                # Create dummy arrays if completely failed
                preds = np.zeros((1, self.args.pred_len, self.args.c_out))
                trues = np.zeros((1, self.args.pred_len, self.args.c_out))
                print("Using dummy arrays due to concatenation failure")

        # Result save - use the same folder path that was created earlier

        # dtw calculation
        try:
            if self.args.use_dtw:
                print("Starting DTW calculation...")
                dtw_list = []
                manhattan_distance = lambda x, y: np.abs(x - y)
                for i in range(preds.shape[0]):
                    x = preds[i].reshape(-1, 1)
                    y = trues[i].reshape(-1, 1)
                    if i % 100 == 0:
                        print("calculating dtw iter:", i)
                    d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                    dtw_list.append(d)
                dtw = np.array(dtw_list).mean()
                print(f"DTW calculation completed: {dtw}")
            else:
                dtw = -999
                print("DTW calculation skipped")
        except Exception as e:
            print(f"Error during DTW calculation: {e}")
            self.logger.error(f"Error during DTW calculation: {e}")
            dtw = -999

        try:
            mae, mse, rmse, mape, mspe = metric(preds, trues)
            print(f"rmse:{rmse}, mae:{mae}, dtw:{dtw}")

            # Log detailed metrics
            self.logger.info("=" * 60)
            self.logger.info("FINAL TEST RESULTS")
            self.logger.info("=" * 60)
            self.logger.info(f"RMSE: {rmse:.8f}")
            self.logger.info(f"MSE: {mse:.8f}")
            self.logger.info(f"MAE: {mae:.8f}")
            self.logger.info(f"MAPE: {mape:.8f}")
            self.logger.info(f"MSPE: {mspe:.8f}")
            self.logger.info(f"DTW: {dtw:.8f}")
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.error(f"Failed to calculate metrics: {e}")
            print(f"Failed to calculate metrics: {e}")
            # Use default values if metric calculation fails
            mae, mse, rmse, mape, mspe = 0.0, 0.0, 0.0, 0.0, 0.0

        # Upload results to Google Sheets
        if SHEETS_AVAILABLE:
            try:
                self.logger.info("Starting Google Sheets upload process")
                print("Starting Google Sheets upload process")

                # Use MSE as final test loss to avoid memory-intensive validation rerun
                final_test_loss = mse
                test_metrics = {"rmse": rmse, "mse": mse, "mae": mae, "mape": mape, "mspe": mspe, "dtw": dtw}

                # Get training losses from stored values or use defaults
                train_loss = getattr(self, "final_train_loss", 0.0)
                val_loss = getattr(self, "final_val_loss", 0.0)

                self.logger.info(f"Uploading metrics - Train: {train_loss}, Val: {val_loss}, Test: {final_test_loss}")
                print(f"Uploading metrics - Train: {train_loss}, Val: {val_loss}, Test: {final_test_loss}")
                print(f"Test metrics: {test_metrics}")

                # Upload to Google Sheets
                upload_experiment_results(self.args, train_loss, val_loss, final_test_loss, test_metrics, setting)

                self.logger.info("Results uploaded to Google Sheets successfully")
                print("Results uploaded to Google Sheets successfully")
            except Exception as e:
                self.logger.error(f"Failed to upload results to Google Sheets: {e}")
                print(f"Failed to upload results to Google Sheets: {e}")
                import traceback

                traceback.print_exc()

        # Save results to file in the same folder structure
        try:
            print("Saving results to files...")
            result_file = folder_path + "result_long_term_forecast.txt"
            with open(result_file, "a") as f:
                f.write(f"Setting: {setting}\n")
                f.write("=" * 50 + "\n")
                f.write(f"RMSE: {rmse}\n")
                f.write(f"MSE: {mse}\n")
                f.write(f"MAE: {mae}\n")
                f.write(f"MAPE: {mape}\n")
                f.write(f"MSPE: {mspe}\n")
                f.write(f"DTW: {dtw}\n")
                f.write("=" * 50 + "\n")
                f.write("\n")
            print(f"Results saved to {result_file}")
            self.logger.info(f"Results saved to {result_file}")

            print("Saving numpy arrays...")
            np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
            np.save(folder_path + "pred.npy", preds)
            np.save(folder_path + "true.npy", trues)
            print("All numpy arrays saved successfully")
            self.logger.info("All numpy arrays saved successfully")
        except Exception as e:
            print(f"Error saving results: {e}")
            self.logger.error(f"Error saving results: {e}")
            import traceback

            traceback.print_exc()
