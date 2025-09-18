"""
Google Sheets uploader for experiment results (Refactored)
Main module that orchestrates uploading to different worksheets with formatting.
"""

import os
import time
from typing import Any

from .sheets_base import GSPREAD_AVAILABLE, BaseSheetsHandler, get_git_info
from .sheets_formatting import SheetsFormatter


class SheetsUploader(BaseSheetsHandler):
    """Main class to handle uploading experiment results to Google Sheets"""

    def __init__(self, spreadsheet_url: str, credentials_path: str | None = None, enable_formatting: bool = True):
        super().__init__(spreadsheet_url, credentials_path)
        self.formatter = SheetsFormatter()
        self.enable_formatting = enable_formatting

    def upload_results(self, args: Any, train_loss: float, val_loss: float, test_loss: float, test_metrics: dict[str, float], setting: str) -> None:
        """Upload experiment results to Google Sheets"""

        print(f"DEBUG: GSPREAD_AVAILABLE = {GSPREAD_AVAILABLE}")
        print(f"DEBUG: self.main_worksheet = {self.main_worksheet}")
        print(f"DEBUG: self.gc = {self.gc}")
        print(f"DEBUG: self.sheet = {self.sheet}")

        if not GSPREAD_AVAILABLE or not self.main_worksheet:
            print("Google Sheets upload skipped (not available)")
            print(f"GSPREAD_AVAILABLE: {GSPREAD_AVAILABLE}")
            print(f"main_worksheet: {self.main_worksheet}")
            self.save_to_local_backup(args, train_loss, val_loss, test_loss, test_metrics, setting)
            return

        task = getattr(args, "task_name", "")

        # Special handling for classification task
        if task == "classification":
            self._upload_classification_result(args, test_metrics, setting)
            return

        # Special handling for short-term forecasting task
        if task == "short_term_forecast":
            self._upload_short_term_forecast_result(args, test_metrics, setting)
            return

        try:
            # Setup headers if needed
            self._setup_main_headers()

            # Get git information
            git_branch, git_commit = get_git_info()

            # Prepare row data
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            trial_name = getattr(args, "trial_name", "") or f"Trial_{timestamp.replace(' ', '_').replace(':', '')}"
            # Use model_id for dataset name (e.g., ECL_96_96) instead of data parameter (custom)
            dataset = getattr(args, "model_id", getattr(args, "data", "Unknown"))

            row_data = [
                trial_name,
                dataset,
                getattr(args, "model", ""),
                f"{train_loss:.8f}",
                f"{val_loss:.8f}",
                f"{test_loss:.8f}",
                f"{test_metrics.get('rmse', 0):.8f}",
                f"{test_metrics.get('mse', 0):.8f}",
                f"{test_metrics.get('mae', 0):.8f}",
                f"{test_metrics.get('mape', 0):.8f}",
                f"{test_metrics.get('mspe', 0):.8f}",
                timestamp,
                str(getattr(args, "seq_len", "")),
                str(getattr(args, "pred_len", "")),
                str(getattr(args, "batch_size", "")),
                str(getattr(args, "learning_rate", "")),
                getattr(args, "des", ""),
                setting,
                git_branch,
                git_commit,
            ]

            print(f"Debug row_data for {trial_name}: length={len(row_data)}")
            print(f"MSPE value: {test_metrics.get('mspe', 'NOT_FOUND')}")
            print(f"Test metrics keys: {list(test_metrics.keys())}")

            # Upload to main LB sheet
            print("DEBUG: About to upload to main LB sheet...")
            success = self.add_row_to_worksheet(self.main_worksheet, row_data)
            print(f"DEBUG: Main LB sheet upload success: {success}")
            print(f"Formatting enabled: {self.enable_formatting}")
            if success and self.enable_formatting:
                # Apply formatting to main worksheet with error handling
                try:
                    print("Attempting to apply main worksheet formatting...")
                    self.formatter.highlight_best_results_by_dataset(self.main_worksheet)
                except Exception as format_error:
                    print(f"Warning: Formatting failed (possibly due to API quota): {format_error}")
                    # Continue execution even if formatting fails
            else:
                print("Skipping main worksheet formatting (disabled or upload failed)")

            # Determine worksheet name based on task
            if task == "long_term_forecast":
                # For long-term forecast, use model_id directly (e.g., ECL_96_96)
                # since it already contains dataset_seq_len_pred_len format
                worksheet_name = dataset
            else:
                worksheet_name = dataset

            # Upload to dataset-specific worksheet
            dataset_worksheet = self.setup_dataset_worksheet(worksheet_name)
            if dataset_worksheet:
                # Prepare row data with settings first, then results
                dataset_row_data = [
                    # Settings columns
                    trial_name,
                    getattr(args, "model", ""),
                    timestamp,
                    str(getattr(args, "seq_len", "")),
                    str(getattr(args, "pred_len", "")),
                    str(getattr(args, "batch_size", "")),
                    str(getattr(args, "learning_rate", "")),
                    getattr(args, "des", ""),
                    setting,
                    git_branch,
                    git_commit,
                    # Results columns
                    f"{train_loss:.8f}",
                    f"{val_loss:.8f}",
                    f"{test_loss:.8f}",
                    f"{test_metrics.get('rmse', 0):.8f}",
                    f"{test_metrics.get('mse', 0):.8f}",
                    f"{test_metrics.get('mae', 0):.8f}",
                    f"{test_metrics.get('mape', 0):.8f}",
                    f"{test_metrics.get('mspe', 0):.8f}",
                ]

                success = self.add_row_to_worksheet(dataset_worksheet, dataset_row_data)
                if success and self.enable_formatting:
                    # Apply dataset formatting (colors and best value highlighting) with error handling
                    try:
                        print("Attempting to apply dataset formatting...")
                        self.formatter.apply_dataset_formatting(dataset_worksheet)
                    except Exception as format_error:
                        print(f"Warning: Dataset formatting failed (possibly due to API quota): {format_error}")
                        # Continue execution even if formatting fails
                else:
                    print("Skipping dataset formatting (disabled or upload failed)")

            # Update LB_ALL view with this trial's data
            self._update_lb_all_trial(trial_name, worksheet_name, row_data)

            print(f"Results uploaded to Google Sheets successfully for trial: {trial_name}")
            print(f"  Dataset: {dataset}")
            print(f"  Git: {git_branch}@{git_commit}")
            print(f"  Test RMSE: {test_metrics.get('rmse', 0):.8f}")
            print(f"  Test MSE: {test_metrics.get('mse', 0):.8f}")
            print(f"  Test MAE: {test_metrics.get('mae', 0):.8f}")

        except Exception as e:
            print(f"Failed to upload results to Google Sheets: {e}")
            # Save to local backup as fallback
            self.save_to_local_backup(args, train_loss, val_loss, test_loss, test_metrics, setting)

    def _upload_classification_result(self, args: Any, test_metrics: dict[str, float], setting: str) -> None:
        """Upload classification result to classification worksheet"""
        try:
            # Get git information
            git_branch, git_commit = get_git_info()

            # Prepare base trial data
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            trial_name = getattr(args, "trial_name", "") or f"Trial_{timestamp.replace(' ', '_').replace(':', '')}"
            dataset_name = getattr(args, "model_id", getattr(args, "data", "Unknown"))
            accuracy = test_metrics.get("accuracy", 0.0)

            # Upload to classification worksheet
            classification_worksheet = self.setup_classification_worksheet()
            if classification_worksheet:
                # Get existing data to check if we need to update or add
                all_data = classification_worksheet.get_all_values()
                headers = all_data[0] if all_data else []

                # Find or create trial row
                trial_row_idx = None
                for i, row in enumerate(all_data[1:], start=2):
                    if len(row) > 0 and row[0] == trial_name:
                        trial_row_idx = i
                        break

                # Prepare base row data
                if not trial_row_idx:
                    # New trial - create full row
                    classification_row_data = [
                        trial_name,
                        getattr(args, "model", ""),
                        timestamp,
                        str(getattr(args, "batch_size", "")),
                        str(getattr(args, "learning_rate", "")),
                        getattr(args, "des", ""),
                        git_branch,
                        git_commit,
                    ]

                    # Add empty values for all dataset accuracy columns
                    dataset_names = [
                        "EthanolConcentration",
                        "FaceDetection",
                        "Handwriting",
                        "Heartbeat",
                        "JapaneseVowels",
                        "PEMS-SF",
                        "SelfRegulationSCP1",
                        "SelfRegulationSCP2",
                        "SpokenArabicDigits",
                        "UWaveGestureLibrary",
                    ]

                    for dataset in dataset_names:
                        if dataset == dataset_name:
                            classification_row_data.append(f"{accuracy:.6f}")
                        else:
                            classification_row_data.append("")

                    # Add new row
                    classification_worksheet.append_row(classification_row_data)
                    print(f"Added new classification trial row for {trial_name}")

                else:
                    # Update existing trial row - just update the specific dataset column
                    dataset_col_name = f"{dataset_name}_Acc"
                    if dataset_col_name in headers:
                        col_idx = headers.index(dataset_col_name) + 1  # +1 for 1-based indexing
                        cell_address = f"{chr(ord('A') + col_idx - 1)}{trial_row_idx}"
                        classification_worksheet.update(cell_address, [[f"{accuracy:.6f}"]])
                        print(f"Updated {dataset_col_name} for trial {trial_name}: {accuracy:.6f}")

            # Also update LB_ALL if this is part of a complete trial
            self._update_lb_all_with_classification(trial_name, dataset_name, accuracy, args, git_branch, git_commit, timestamp)

            print(f"Classification result uploaded successfully for {dataset_name} in trial {trial_name}")
            print(f"  Accuracy: {accuracy:.6f}")

        except Exception as e:
            print(f"Failed to upload classification result: {e}")
            import traceback

            traceback.print_exc()

    def _upload_short_term_forecast_result(self, args: Any, test_metrics: dict[str, float], setting: str) -> None:
        """Upload short-term forecasting result to short-term-forecasting worksheet"""
        try:
            # Get git information
            git_branch, git_commit = get_git_info()

            # Prepare base trial data
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            trial_name = getattr(args, "trial_name", "") or f"Trial_{timestamp.replace(' ', '_').replace(':', '')}"
            seasonal_pattern = getattr(args, "seasonal_patterns", "Unknown")
            smape = test_metrics.get("smape", 0.0)
            mase = test_metrics.get("mase", 0.0)
            owa = test_metrics.get("owa", 0.0)

            # Upload to short-term-forecasting worksheet
            stf_worksheet = self.setup_short_term_forecast_worksheet()
            if stf_worksheet:
                # Get existing data to check if we need to update or add
                all_data = stf_worksheet.get_all_values()
                headers = all_data[0] if all_data else []

                # Find or create trial row
                trial_row_idx = None
                for i, row in enumerate(all_data[1:], start=2):
                    if len(row) > 0 and row[0] == trial_name:
                        trial_row_idx = i
                        break

                # Prepare base row data
                if not trial_row_idx:
                    # New trial - create full row
                    stf_row_data = [
                        trial_name,
                        getattr(args, "model", ""),
                        timestamp,
                        str(getattr(args, "batch_size", "")),
                        str(getattr(args, "learning_rate", "")),
                        getattr(args, "des", ""),
                        git_branch,
                        git_commit,
                    ]

                    # Add empty values for all seasonal pattern metric columns
                    seasonal_patterns = ["Monthly", "Yearly", "Quarterly", "Weekly", "Daily", "Hourly"]

                    for pattern in seasonal_patterns:
                        if pattern == seasonal_pattern:
                            stf_row_data.extend([f"{smape:.6f}", f"{mase:.6f}", f"{owa:.6f}"])
                        else:
                            stf_row_data.extend(["", "", ""])

                    # Add new row
                    stf_worksheet.append_row(stf_row_data)
                    print(f"Added new short-term forecasting trial row for {trial_name}")

                else:
                    # Update existing trial row - just update the specific seasonal pattern columns
                    smape_col_name = f"{seasonal_pattern}_SMAPE"
                    mase_col_name = f"{seasonal_pattern}_MASE"
                    owa_col_name = f"{seasonal_pattern}_OWA"

                    # Get current row data to properly update
                    current_row_data = all_data[trial_row_idx - 1] if trial_row_idx <= len(all_data) else []

                    # Extend row_data if needed to match headers length
                    while len(current_row_data) < len(headers):
                        current_row_data.append("")

                    # Update the seasonal pattern metrics
                    if smape_col_name in headers:
                        col_idx = headers.index(smape_col_name)
                        current_row_data[col_idx] = f"{smape:.6f}"

                    if mase_col_name in headers:
                        col_idx = headers.index(mase_col_name)
                        current_row_data[col_idx] = f"{mase:.6f}"

                    if owa_col_name in headers:
                        col_idx = headers.index(owa_col_name)
                        current_row_data[col_idx] = f"{owa:.6f}"

                    # Update the entire row at once
                    stf_worksheet.update(f"A{trial_row_idx}:{self._get_column_letter(len(current_row_data) - 1)}{trial_row_idx}", [current_row_data])
                    print(f"Updated {seasonal_pattern} metrics for trial {trial_name}")

            # Also update LB_ALL
            self._update_lb_all_with_short_term_forecast(trial_name, seasonal_pattern, smape, mase, owa, args, git_branch, git_commit, timestamp)

            print(f"Short-term forecasting result uploaded successfully for {seasonal_pattern} in trial {trial_name}")
            print(f"  SMAPE: {smape:.6f}, MASE: {mase:.6f}, OWA: {owa:.6f}")

        except Exception as e:
            print(f"Failed to upload short-term forecasting result: {e}")
            import traceback

            traceback.print_exc()

    def _update_lb_all_trial(self, trial_name: str, worksheet_name: str, row_data: list[str]) -> None:
        """Update specific trial in LB_ALL worksheet"""
        try:
            lb_all_worksheet = self.setup_lb_all_worksheet()
            if not lb_all_worksheet:
                return

            # Get current LB_ALL data
            all_data = lb_all_worksheet.get_all_values()

            # Define base headers in desired order
            base_headers = [
                "Trial Name",
                "Model",
                "Timestamp",
                "Seq Len",
                "Pred Len",
                "Batch Size",
                "Learning Rate",
                "Description",
                "Setting",
                "Git Branch",
                "Git Commit",
            ]

            # Setup headers if empty
            if len(all_data) == 0:
                lb_all_worksheet.update("1:1", [base_headers])
                lb_all_worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})
                all_data = [base_headers]

            headers = all_data[0] if all_data else []

            # Find existing trial row with same trial_name (don't make unique, update instead)
            trial_row_idx = None
            for i, row in enumerate(all_data[1:], start=2):  # Start from row 2 (after headers)
                if len(row) > 0 and row[0] == trial_name:
                    trial_row_idx = i
                    break

            # Prepare base data from row_data
            base_data = {
                "Trial Name": trial_name,
                "Model": row_data[2] if len(row_data) > 2 else "",
                "Timestamp": row_data[11] if len(row_data) > 11 else "",
                "Seq Len": row_data[12] if len(row_data) > 12 else "",
                "Pred Len": row_data[13] if len(row_data) > 13 else "",
                "Batch Size": row_data[14] if len(row_data) > 14 else "",
                "Learning Rate": row_data[15] if len(row_data) > 15 else "",
                "Description": row_data[16] if len(row_data) > 16 else "",
                "Setting": row_data[17] if len(row_data) > 17 else "",
                "Git Branch": row_data[18] if len(row_data) > 18 else "",
                "Git Commit": row_data[19] if len(row_data) > 19 else "",
            }

            # Add dataset-specific columns
            metric_names = ["Train Loss", "Val Loss", "Test Loss", "Test RMSE", "Test MSE", "Test MAE", "Test MAPE", "Test MSPE"]
            dataset_metrics = {}

            for i, metric in enumerate(metric_names):
                col_name = f"{worksheet_name}_{metric.replace(' ', '_')}"
                if i + 3 < len(row_data):  # Train Loss starts at index 3
                    dataset_metrics[col_name] = row_data[i + 3]
                    print(f"Added metric {col_name}: {row_data[i + 3]}")
                else:
                    print(f"Warning: Missing data for metric {col_name} at index {i + 3}, row_data length: {len(row_data)}")

            # Ensure base headers are in correct order
            new_headers = base_headers.copy()

            # Add existing dataset columns from current headers
            dataset_columns = [h for h in headers if h not in base_headers]
            new_headers.extend(dataset_columns)

            # Check if we need to add new dataset columns
            for col_name in dataset_metrics.keys():
                if col_name not in new_headers:
                    new_headers.append(col_name)

            # Update headers if structure changed
            headers_changed = new_headers != headers
            if headers_changed:
                lb_all_worksheet.update("1:1", [new_headers])
                headers = new_headers

            # Prepare row data
            updated_row = [""] * len(headers)

            # Fill base data
            for key, value in base_data.items():
                if key in headers:
                    idx = headers.index(key)
                    updated_row[idx] = value

            # Fill dataset-specific data
            for key, value in dataset_metrics.items():
                if key in headers:
                    idx = headers.index(key)
                    updated_row[idx] = value

                    # Update or add the row
            if trial_row_idx:
                # Update existing row - batch update for efficiency
                # Prepare update data only for columns that need updating
                updates_to_make = []

                # Collect base data updates
                for key, value in base_data.items():
                    if key in headers:
                        col_idx = headers.index(key)
                        col_letter = self._get_column_letter(col_idx)
                        cell_address = f"{col_letter}{trial_row_idx}"
                        updates_to_make.append((cell_address, value))

                # Collect dataset-specific metrics updates
                for key, value in dataset_metrics.items():
                    if key in headers:
                        col_idx = headers.index(key)
                        col_letter = self._get_column_letter(col_idx)
                        cell_address = f"{col_letter}{trial_row_idx}"
                        updates_to_make.append((cell_address, value))

                # Perform batch updates
                try:
                    print(f"Performing {len(updates_to_make)} updates for trial {trial_name}")
                    for cell_address, value in updates_to_make:
                        print(f"  Updating {cell_address} with value: {value}")
                        lb_all_worksheet.update(cell_address, [[value]])
                    print(f"Successfully updated {len(updates_to_make)} cells for trial {trial_name}")
                except Exception as e:
                    print(f"Failed to batch update trial {trial_name}: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                # Add new row
                lb_all_worksheet.append_row(updated_row)

            print(f"LB_ALL updated for trial: {trial_name}, worksheet: {worksheet_name}")

            # Apply dataset colors only to new columns to avoid overwriting existing formatting
            new_columns = list(dataset_metrics.keys())
            if new_columns:
                print(f"Applying selective formatting to new columns: {new_columns}")
                self.formatter.apply_dataset_colors_selective(lb_all_worksheet, headers, new_columns)

                print(f"Highlighting best values only for new columns: {new_columns}")
                self.formatter.highlight_best_values_in_lb_all(lb_all_worksheet, new_columns)
            else:
                print("No new columns to format or highlight")

        except Exception as e:
            print(f"Failed to update LB_ALL trial {trial_name} for worksheet {worksheet_name}: {e}")

    def _update_lb_all_with_classification(
        self, trial_name: str, dataset_name: str, accuracy: float, args: Any, git_branch: str, git_commit: str, timestamp: str
    ) -> None:
        """Update LB_ALL worksheet with individual classification dataset result"""
        try:
            lb_all_worksheet = self.setup_lb_all_worksheet()
            if not lb_all_worksheet:
                return

            # Get current LB_ALL data
            all_data = lb_all_worksheet.get_all_values()

            # Define base headers
            base_headers = [
                "Trial Name",
                "Model",
                "Timestamp",
                "Seq Len",
                "Pred Len",
                "Batch Size",
                "Learning Rate",
                "Description",
                "Setting",
                "Git Branch",
                "Git Commit",
            ]

            # Setup headers if empty
            if len(all_data) == 0:
                print("Setting up LB_ALL worksheet headers...")
                lb_all_worksheet.update("1:1", [base_headers])
                lb_all_worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})
                all_data = [base_headers]
                print("LB_ALL worksheet headers created")

            headers = all_data[0] if all_data else []

            # Find existing trial row with same trial_name (don't make unique, update instead)
            trial_row_idx = None
            for i, row in enumerate(all_data[1:], start=2):
                if len(row) > 0 and row[0] == trial_name:
                    trial_row_idx = i
                    break

            # Add dataset accuracy column header if not exists
            dataset_col_name = f"{dataset_name}_Acc"
            if dataset_col_name not in headers:
                headers.append(dataset_col_name)
                lb_all_worksheet.update("1:1", [headers])
                lb_all_worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})
                print(f"Added {dataset_col_name} column to LB_ALL headers")

            # Find column index for this dataset
            col_idx = headers.index(dataset_col_name)

            if not trial_row_idx:
                # Create new trial row with proper length and positioning
                row_data = [""] * len(headers)  # Initialize with empty strings for all columns

                # Fill base data in correct positions
                row_data[headers.index("Trial Name")] = trial_name
                row_data[headers.index("Model")] = getattr(args, "model", "")
                row_data[headers.index("Timestamp")] = timestamp
                row_data[headers.index("Seq Len")] = ""  # Not applicable for classification
                row_data[headers.index("Pred Len")] = ""  # Not applicable for classification
                row_data[headers.index("Batch Size")] = str(getattr(args, "batch_size", ""))
                row_data[headers.index("Learning Rate")] = str(getattr(args, "learning_rate", ""))
                row_data[headers.index("Description")] = getattr(args, "des", "")
                row_data[headers.index("Setting")] = "classification"
                row_data[headers.index("Git Branch")] = git_branch
                row_data[headers.index("Git Commit")] = git_commit

                # Set accuracy in the correct dataset column
                row_data[col_idx] = f"{accuracy:.6f}"

                lb_all_worksheet.append_row(row_data)
                print(f"Added new trial {trial_name} to LB_ALL with {dataset_name} accuracy at column {col_idx + 1}")
            else:
                # Update existing trial row - just update the dataset accuracy column
                cell_address = f"{chr(ord('A') + col_idx)}{trial_row_idx}"
                lb_all_worksheet.update(cell_address, [[f"{accuracy:.6f}"]])
                print(f"Updated LB_ALL {dataset_name} accuracy for trial {trial_name} at column {col_idx + 1}")

            # Apply dataset colors only to new columns to avoid overwriting existing formatting
            self.formatter.apply_dataset_colors_selective(lb_all_worksheet, headers, [dataset_col_name])

            # Only highlight best values for the accuracy column
            new_columns = [dataset_col_name]
            print(f"Highlighting best values only for classification column: {new_columns}")
            self.formatter.highlight_best_values_in_lb_all(lb_all_worksheet, new_columns)

        except Exception as e:
            print(f"Failed to update LB_ALL with classification result: {e}")
            import traceback

            traceback.print_exc()

    def _update_lb_all_with_short_term_forecast(
        self, trial_name: str, seasonal_pattern: str, smape: float, mase: float, owa: float, args: Any, git_branch: str, git_commit: str, timestamp: str
    ) -> None:
        """Update LB_ALL worksheet with individual short-term forecasting seasonal pattern result"""
        try:
            lb_all_worksheet = self.setup_lb_all_worksheet()
            if not lb_all_worksheet:
                return

            # Get current LB_ALL data
            all_data = lb_all_worksheet.get_all_values()

            # Define base headers
            base_headers = [
                "Trial Name",
                "Model",
                "Timestamp",
                "Seq Len",
                "Pred Len",
                "Batch Size",
                "Learning Rate",
                "Description",
                "Setting",
                "Git Branch",
                "Git Commit",
            ]

            # Setup headers if empty
            if len(all_data) == 0:
                print("Setting up LB_ALL worksheet headers...")
                lb_all_worksheet.update("1:1", [base_headers])
                lb_all_worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})
                all_data = [base_headers]
                print("LB_ALL worksheet headers created")

            headers = all_data[0] if all_data else []

            # Find existing trial row with same trial_name (don't make unique, update instead)
            trial_row_idx = None
            for i, row in enumerate(all_data[1:], start=2):
                if len(row) > 0 and row[0] == trial_name:
                    trial_row_idx = i
                    break

            # Add seasonal pattern metric columns headers if not exist
            smape_col_name = f"{seasonal_pattern}_SMAPE"
            mase_col_name = f"{seasonal_pattern}_MASE"
            owa_col_name = f"{seasonal_pattern}_OWA"

            headers_changed = False
            for col_name in [smape_col_name, mase_col_name, owa_col_name]:
                if col_name not in headers:
                    headers.append(col_name)
                    headers_changed = True

            if headers_changed:
                lb_all_worksheet.update("1:1", [headers])
                lb_all_worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})
                print(f"Added {seasonal_pattern} metric columns to LB_ALL headers")

            if not trial_row_idx:
                # Create new trial row with proper length and positioning
                row_data = [""] * len(headers)  # Initialize with empty strings for all columns

                # Fill base data in correct positions
                row_data[headers.index("Trial Name")] = trial_name
                row_data[headers.index("Model")] = getattr(args, "model", "")
                row_data[headers.index("Timestamp")] = timestamp
                row_data[headers.index("Seq Len")] = ""  # Not applicable for short-term forecasting
                row_data[headers.index("Pred Len")] = ""  # Not applicable for short-term forecasting
                row_data[headers.index("Batch Size")] = str(getattr(args, "batch_size", ""))
                row_data[headers.index("Learning Rate")] = str(getattr(args, "learning_rate", ""))
                row_data[headers.index("Description")] = getattr(args, "des", "")
                row_data[headers.index("Setting")] = "short_term_forecast"
                row_data[headers.index("Git Branch")] = git_branch
                row_data[headers.index("Git Commit")] = git_commit

                # Set metrics in the correct seasonal pattern columns
                row_data[headers.index(smape_col_name)] = f"{smape:.6f}"
                row_data[headers.index(mase_col_name)] = f"{mase:.6f}"
                row_data[headers.index(owa_col_name)] = f"{owa:.6f}"

                lb_all_worksheet.append_row(row_data)
                print(f"Added new trial {trial_name} to LB_ALL with {seasonal_pattern} metrics")
            else:
                # Update existing trial row - just update the seasonal pattern metric columns
                current_row_data = all_data[trial_row_idx - 1] if trial_row_idx <= len(all_data) else []

                # Extend row_data if needed to match headers length
                while len(current_row_data) < len(headers):
                    current_row_data.append("")

                # Update only the seasonal pattern metrics
                for col_name, value in [(smape_col_name, smape), (mase_col_name, mase), (owa_col_name, owa)]:
                    col_idx = headers.index(col_name)
                    current_row_data[col_idx] = f"{value:.6f}"

                # Update the entire row at once to avoid cell-by-cell conflicts
                lb_all_worksheet.update(f"A{trial_row_idx}:{self._get_column_letter(len(current_row_data) - 1)}{trial_row_idx}", [current_row_data])
                print(f"Updated LB_ALL {seasonal_pattern} metrics for trial {trial_name}")

            # Apply dataset colors only to new columns to avoid overwriting existing formatting
            self.formatter.apply_dataset_colors_selective(lb_all_worksheet, headers, [smape_col_name, mase_col_name, owa_col_name])

            # Only highlight best values for the seasonal pattern metric columns
            new_columns = [smape_col_name, mase_col_name, owa_col_name]
            print(f"Highlighting best values only for short-term forecast columns: {new_columns}")
            self.formatter.highlight_best_values_in_lb_all(lb_all_worksheet, new_columns)

        except Exception as e:
            print(f"Failed to update LB_ALL with short-term forecasting result: {e}")
            import traceback

            traceback.print_exc()

    def setup_classification_worksheet(self) -> Any:
        """Setup or get classification worksheet"""
        if not self.sheet:
            return None

        try:
            # Try to get existing classification worksheet
            classification_worksheet = self.sheet.worksheet("classification")

            # Check if headers exist and are complete
            try:
                existing_headers = classification_worksheet.row_values(1)
                expected_headers = [
                    "Trial Name",
                    "Model",
                    "Timestamp",
                    "Batch Size",
                    "Learning Rate",
                    "Description",
                    "Git Branch",
                    "Git Commit",
                    "EthanolConcentration_Acc",
                    "FaceDetection_Acc",
                    "Handwriting_Acc",
                    "Heartbeat_Acc",
                    "JapaneseVowels_Acc",
                    "PEMS-SF_Acc",
                    "SelfRegulationSCP1_Acc",
                    "SelfRegulationSCP2_Acc",
                    "SpokenArabicDigits_Acc",
                    "UWaveGestureLibrary_Acc",
                ]

                # If headers are missing or incomplete, update them
                if not existing_headers or len(existing_headers) < len(expected_headers) or existing_headers[: len(expected_headers)] != expected_headers:
                    print("Setting up classification worksheet headers...")
                    classification_worksheet.update("1:1", [expected_headers])
                    classification_worksheet.format(
                        "1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}}
                    )
                    print("Classification worksheet headers updated")

            except Exception as e:
                print(f"Error checking/updating classification headers: {e}")

        except:
            # Create new classification worksheet
            print("Creating new classification worksheet...")
            classification_worksheet = self.sheet.add_worksheet(title="classification", rows=1000, cols=25)

            # Setup headers for classification worksheet
            headers = [
                "Trial Name",
                "Model",
                "Timestamp",
                "Batch Size",
                "Learning Rate",
                "Description",
                "Git Branch",
                "Git Commit",
                "EthanolConcentration_Acc",
                "FaceDetection_Acc",
                "Handwriting_Acc",
                "Heartbeat_Acc",
                "JapaneseVowels_Acc",
                "PEMS-SF_Acc",
                "SelfRegulationSCP1_Acc",
                "SelfRegulationSCP2_Acc",
                "SpokenArabicDigits_Acc",
                "UWaveGestureLibrary_Acc",
            ]

            classification_worksheet.update("1:1", [headers])
            classification_worksheet.format(
                "1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}}
            )
            print("Classification worksheet created with headers")

        return classification_worksheet

    def setup_short_term_forecast_worksheet(self) -> Any:
        """Setup or get short-term-forecasting worksheet"""
        if not self.sheet:
            return None

        try:
            # Try to get existing short-term-forecasting worksheet
            stf_worksheet = self.sheet.worksheet("short-term-forecasting")

            # Check if headers exist and are complete
            try:
                existing_headers = stf_worksheet.row_values(1)
                expected_headers = [
                    "Trial Name",
                    "Model",
                    "Timestamp",
                    "Batch Size",
                    "Learning Rate",
                    "Description",
                    "Git Branch",
                    "Git Commit",
                ]

                # Add all seasonal pattern metric columns
                seasonal_patterns = ["Monthly", "Yearly", "Quarterly", "Weekly", "Daily", "Hourly"]
                for pattern in seasonal_patterns:
                    expected_headers.extend([f"{pattern}_SMAPE", f"{pattern}_MASE", f"{pattern}_OWA"])

                # If headers are missing or incomplete, update them
                if not existing_headers or len(existing_headers) < len(expected_headers) or existing_headers[: len(expected_headers)] != expected_headers:
                    print("Setting up short-term-forecasting worksheet headers...")
                    stf_worksheet.update("1:1", [expected_headers])
                    stf_worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})
                    print("Short-term-forecasting worksheet headers updated")

            except Exception as e:
                print(f"Error checking/updating short-term-forecasting headers: {e}")

        except:
            # Create new short-term-forecasting worksheet
            print("Creating new short-term-forecasting worksheet...")
            stf_worksheet = self.sheet.add_worksheet(title="short-term-forecasting", rows=1000, cols=30)

            # Setup headers for short-term-forecasting worksheet
            headers = [
                "Trial Name",
                "Model",
                "Timestamp",
                "Batch Size",
                "Learning Rate",
                "Description",
                "Git Branch",
                "Git Commit",
            ]

            # Add all seasonal pattern metric columns
            seasonal_patterns = ["Monthly", "Yearly", "Quarterly", "Weekly", "Daily", "Hourly"]
            for pattern in seasonal_patterns:
                headers.extend([f"{pattern}_SMAPE", f"{pattern}_MASE", f"{pattern}_OWA"])

            stf_worksheet.update("1:1", [headers])
            stf_worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})
            print("Short-term-forecasting worksheet created with headers")

        return stf_worksheet

    def _get_column_letter(self, col_idx: int) -> str:
        """Convert column index to Excel-style column letter (A, B, ..., Z, AA, AB, ...)"""
        result = ""
        while col_idx >= 0:
            result = chr(ord("A") + (col_idx % 26)) + result
            col_idx = col_idx // 26 - 1
            if col_idx < 0:
                break
        return result


# Global instance
_sheets_uploader = None


def get_sheets_uploader() -> SheetsUploader | None:
    """Get global sheets uploader instance"""
    global _sheets_uploader  # noqa: PLW0603

    if _sheets_uploader is None:
        spreadsheet_url = "https://docs.google.com/spreadsheets/d/1myE0Xnj5eogFMdfMeDWXF4CA7d6b-tgpEgSW0o6UPgA/edit?usp=sharing"
        # Try to find service account credentials
        possible_paths = ["/home/vscode/.config/gspread/service_account.json", "~/.config/gspread/service_account.json", "service_account.json"]

        credentials_path = None
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                credentials_path = expanded_path
                break

        # Check environment variable to disable formatting if needed
        enable_formatting = os.getenv("SHEETS_ENABLE_FORMATTING", "true").lower() == "true"
        _sheets_uploader = SheetsUploader(spreadsheet_url, credentials_path, enable_formatting)

    return _sheets_uploader


def upload_experiment_results(args: Any, train_loss: float, val_loss: float, test_loss: float, test_metrics: dict[str, float], setting: str) -> None:
    """Convenience function to upload experiment results"""
    uploader = get_sheets_uploader()
    if uploader:
        uploader.upload_results(args, train_loss, val_loss, test_loss, test_metrics, setting)
