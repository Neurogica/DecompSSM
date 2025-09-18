"""
Google Sheets uploader for experiment results
Automatically uploads training and test results to a shared spreadsheet.
Format: trial_name, dataset, train_loss, val_loss, test_loss, test_rmse, mse, mae, mape, mspe
"""

import json
import os
import subprocess
import time
from typing import Any

try:
    import gspread
    from google.oauth2.service_account import Credentials

    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("Warning: gspread not available. Google Sheets upload will be disabled.")


def get_git_info() -> tuple[str, str]:
    """Get current git branch and commit hash"""
    try:
        # Get current branch
        branch_result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True)
        branch = branch_result.stdout.strip()

        # Get current commit hash (short)
        commit_result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        commit = commit_result.stdout.strip()

        return branch, commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", "unknown"


class SheetsUploader:
    """Class to handle uploading experiment results to Google Sheets"""

    def __init__(self, spreadsheet_url: str, credentials_path: str | None = None):
        self.spreadsheet_url = spreadsheet_url
        self.credentials_path = credentials_path
        self.gc: Any = None
        self.sheet: Any = None
        self.worksheet: Any = None

        if GSPREAD_AVAILABLE:
            self._initialize_connection()

    def _initialize_connection(self) -> None:
        """Initialize connection to Google Sheets"""
        try:
            # Try to use credentials from environment or file
            if self.credentials_path and os.path.exists(self.credentials_path):
                # Use service account credentials
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                creds = Credentials.from_service_account_file(self.credentials_path, scopes=scope)
                self.gc = gspread.authorize(creds)
            else:
                # Try using default credentials or environment variable
                self.gc = gspread.service_account()

            # Extract spreadsheet ID from URL
            spreadsheet_id = self._extract_spreadsheet_id(self.spreadsheet_url)
            self.sheet = self.gc.open_by_key(spreadsheet_id)

            # Use 'LB' worksheet or create one
            try:
                self.worksheet = self.sheet.worksheet("LB")
            except:
                self.worksheet = self.sheet.add_worksheet(title="LB", rows=1000, cols=25)

            print(f"Successfully connected to Google Sheets: {self.sheet.title}")

        except Exception as e:
            print(f"Failed to initialize Google Sheets connection: {e}")
            print("Results will only be saved locally.")
            self.gc = None

    def _extract_spreadsheet_id(self, url: str) -> str:
        """Extract spreadsheet ID from Google Sheets URL"""
        if "/spreadsheets/d/" in url:
            return url.split("/spreadsheets/d/")[1].split("/")[0]
        else:
            return url  # Assume it's already an ID

    def _setup_headers(self) -> None:
        """Setup column headers if they don't exist"""
        if not self.worksheet:
            return

        try:
            # Check if headers already exist
            existing_headers = self.worksheet.row_values(1)
            if existing_headers and len(existing_headers) > 1:
                return

            # Define headers with dataset grouping consideration
            headers = [
                "Trial Name",
                "Dataset",
                "Model",
                "Train Loss",
                "Val Loss",
                "Test Loss",
                "Test RMSE",
                "Test MSE",
                "Test MAE",
                "Test MAPE",
                "Test MSPE",
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

            # Set headers
            self.worksheet.update("1:1", [headers])

            # Format headers (bold with background color)
            self.worksheet.format("1:1", {"textFormat": {"bold": True}, "backgroundColor": {"red": 0.8, "green": 0.9, "blue": 1.0}})

            print("Headers setup completed")

        except Exception as e:
            print(f"Failed to setup headers: {e}")

    def _setup_dataset_worksheet(self, dataset_name: str) -> Any:
        """Setup dataset-specific worksheet"""
        if not self.gc or not self.sheet:
            return None

        try:
            # Try to get existing dataset worksheet
            try:
                dataset_worksheet = self.sheet.worksheet(dataset_name)
            except:
                # Create new dataset worksheet
                dataset_worksheet = self.sheet.add_worksheet(title=dataset_name, rows=1000, cols=25)

            # Setup headers for dataset worksheet (same as LB but no Dataset column)
            headers = [
                "Trial Name",
                "Model",
                "Train Loss",
                "Val Loss",
                "Test Loss",
                "Test RMSE",
                "Test MSE",
                "Test MAE",
                "Test MAPE",
                "Test MSPE",
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

            # Check if headers already exist
            existing_headers = dataset_worksheet.row_values(1)
            if not existing_headers or len(existing_headers) <= 1:
                # Set headers
                dataset_worksheet.update("1:1", [headers])
                # Format headers
                dataset_worksheet.format("1:1", {"textFormat": {"bold": True}, "backgroundColor": {"red": 0.8, "green": 0.9, "blue": 1.0}})

            return dataset_worksheet

        except Exception as e:
            print(f"Failed to setup dataset worksheet {dataset_name}: {e}")
            return None

    def _setup_lb_all_worksheet(self) -> Any:
        """Setup LB_ALL worksheet with trial-centric view"""
        if not self.gc or not self.sheet:
            return None

        try:
            # Try to get existing LB_ALL worksheet
            try:
                lb_all_worksheet = self.sheet.worksheet("LB_ALL")
            except:
                # Create new LB_ALL worksheet
                lb_all_worksheet = self.sheet.add_worksheet(title="LB_ALL", rows=1000, cols=50)

            return lb_all_worksheet

        except Exception as e:
            print(f"Failed to setup LB_ALL worksheet: {e}")
            return None

    def _get_dataset_color(self, dataset_index: int) -> dict[str, dict[str, float]]:
        """Get background color for dataset columns"""
        # Define a palette of distinct colors for different datasets
        colors = [
            {"red": 0.95, "green": 0.85, "blue": 0.85},  # Light red
            {"red": 0.85, "green": 0.95, "blue": 0.85},  # Light green
            {"red": 0.85, "green": 0.85, "blue": 0.95},  # Light blue
            {"red": 0.95, "green": 0.95, "blue": 0.85},  # Light yellow
            {"red": 0.95, "green": 0.85, "blue": 0.95},  # Light magenta
            {"red": 0.85, "green": 0.95, "blue": 0.95},  # Light cyan
            {"red": 0.92, "green": 0.88, "blue": 0.85},  # Light peach
            {"red": 0.88, "green": 0.85, "blue": 0.92},  # Light lavender
        ]
        return {"backgroundColor": colors[dataset_index % len(colors)]}

    def _apply_dataset_colors(self, worksheet: Any, headers: list[str]) -> None:
        """Apply colors to dataset-specific columns"""
        try:
            # Identify unique datasets and their columns
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

            # Group columns by dataset
            dataset_columns: dict[str, list[int]] = {}
            for i, header in enumerate(headers):
                if header not in base_headers and "_" in header:
                    dataset_name = header.split("_")[0]
                    if dataset_name not in dataset_columns:
                        dataset_columns[dataset_name] = []
                    dataset_columns[dataset_name].append(i)

            # Apply colors to each dataset's columns
            dataset_index = 0
            for dataset_name, column_indices in sorted(dataset_columns.items()):
                color_format = self._get_dataset_color(dataset_index)
                for col_idx in column_indices:
                    # Format entire column (header + data)
                    col_letter = chr(ord("A") + col_idx)
                    # Format header specifically
                    header_format = {"textFormat": {"bold": True}, "backgroundColor": color_format["backgroundColor"]}
                    worksheet.format(f"{col_letter}1", header_format)

                    # Format data rows with lighter color
                    data_color = color_format["backgroundColor"].copy()
                    # Make data rows slightly lighter
                    for color_key in data_color:
                        data_color[color_key] = min(1.0, data_color[color_key] + 0.05)
                    worksheet.format(f"{col_letter}2:{col_letter}", {"backgroundColor": data_color})

                dataset_index += 1

        except Exception as e:
            print(f"Failed to apply dataset colors: {e}")

    def _highlight_best_values_in_lb_all(self, worksheet: Any) -> None:
        """Highlight best values (lowest) within each metric column with bold text"""
        try:
            # Get all data
            all_data = worksheet.get_all_values()
            if len(all_data) <= 1:  # Only headers or empty
                return

            headers = all_data[0]
            data_rows = all_data[1:]

            # Skip if no data rows
            if not data_rows:
                return

            # Define base headers to exclude from highlighting
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

            # Process each metric column
            for col_idx, header in enumerate(headers):
                # Skip base headers - only process dataset metric columns
                if header in base_headers or "_" not in header:
                    continue

                # Extract values from this column
                values = []
                row_indices = []

                for row_idx, row in enumerate(data_rows):
                    if col_idx < len(row) and row[col_idx].strip():
                        try:
                            # Parse the value
                            val = float(row[col_idx])
                            values.append(val)
                            row_indices.append(row_idx)
                        except (ValueError, TypeError):
                            continue

                # Skip if no valid values
                if not values:
                    continue

                # Find minimum value (best for all metrics)
                min_val = min(values)

                # Apply bold formatting to cells with best value
                col_letter = chr(ord("A") + col_idx)

                # First, remove bold from all cells in this column
                for i in range(len(data_rows)):
                    cell_ref = f"{col_letter}{i + 2}"  # +2 because row 1 is header
                    try:
                        # Get current cell format
                        cell_format = worksheet.get(cell_ref)
                        if cell_format and len(cell_format) > 0 and len(cell_format[0]) > 0:
                            current_value = cell_format[0][0]
                            # Update only text format to remove bold
                            worksheet.format(cell_ref, {"textFormat": {"bold": False}})
                    except:
                        pass

                # Then apply bold to cells with minimum value
                for i, val in enumerate(values):
                    if abs(val - min_val) < 1e-10:  # Use small epsilon for float comparison
                        row_num = row_indices[i] + 2  # +2 because header is row 1
                        cell_ref = f"{col_letter}{row_num}"
                        worksheet.format(cell_ref, {"textFormat": {"bold": True}})

            print("Best values highlighted in LB_ALL worksheet")

        except Exception as e:
            print(f"Failed to highlight best values in LB_ALL: {e}")

    def _highlight_best_values_in_dataset_worksheet(self, worksheet: Any) -> None:
        """Highlight best values (lowest) within each metric column in dataset-specific worksheet"""
        try:
            # Get all data
            all_data = worksheet.get_all_values()
            if len(all_data) <= 1:  # Only headers or empty
                return

            headers = all_data[0]
            data_rows = all_data[1:]

            # Skip if no data rows
            if not data_rows:
                return

            # Define metric columns to highlight (excluding base info columns)
            metric_columns = ["Train Loss", "Val Loss", "Test Loss", "Test RMSE", "Test MSE", "Test MAE", "Test MAPE", "Test MSPE"]

            # Process each metric column
            for metric_name in metric_columns:
                if metric_name not in headers:
                    continue

                col_idx = headers.index(metric_name)

                # Extract values from this column
                values = []
                row_indices = []

                for row_idx, row in enumerate(data_rows):
                    if col_idx < len(row) and row[col_idx].strip():
                        try:
                            # Parse the value
                            val = float(row[col_idx])
                            values.append(val)
                            row_indices.append(row_idx)
                        except (ValueError, TypeError):
                            continue

                # Skip if no valid values
                if not values:
                    continue

                # Find minimum value (best for all metrics)
                min_val = min(values)

                # Apply bold formatting to cells with best value
                col_letter = chr(ord("A") + col_idx)

                # First, remove bold from all cells in this column
                for i in range(len(data_rows)):
                    cell_ref = f"{col_letter}{i + 2}"  # +2 because row 1 is header
                    try:
                        worksheet.format(cell_ref, {"textFormat": {"bold": False}})
                    except:
                        pass

                # Then apply bold to cells with minimum value
                for i, val in enumerate(values):
                    if abs(val - min_val) < 1e-10:  # Use small epsilon for float comparison
                        row_num = row_indices[i] + 2  # +2 because header is row 1
                        cell_ref = f"{col_letter}{row_num}"
                        worksheet.format(cell_ref, {"textFormat": {"bold": True}})

            print("Best values highlighted in dataset worksheet")

        except Exception as e:
            print(f"Failed to highlight best values in dataset worksheet: {e}")

    def _update_lb_all_trial(self, trial_name: str, dataset: str, row_data: list[str]) -> None:
        """Update specific trial in LB_ALL worksheet"""
        try:
            lb_all_worksheet = self._setup_lb_all_worksheet()
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
                lb_all_worksheet.format("1:1", {"textFormat": {"bold": True}, "backgroundColor": {"red": 0.8, "green": 0.9, "blue": 1.0}})
                all_data = [base_headers]

            headers = all_data[0] if all_data else []

            # Find trial row or determine where to add new trial
            trial_row_idx = None
            for i, row in enumerate(all_data[1:], start=2):  # Start from row 2 (after headers)
                if len(row) > 0 and row[0] == trial_name:
                    trial_row_idx = i
                    break

            # Prepare base data from row_data
            base_data = {
                "Trial Name": trial_name,
                "Model": row_data[2] if len(row_data) > 2 else "",  # Model is at index 2 in original row
                "Timestamp": row_data[11] if len(row_data) > 11 else "",  # Timestamp
                "Seq Len": row_data[12] if len(row_data) > 12 else "",
                "Pred Len": row_data[13] if len(row_data) > 13 else "",
                "Batch Size": row_data[14] if len(row_data) > 14 else "",
                "Learning Rate": row_data[15] if len(row_data) > 15 else "",
                "Description": row_data[16] if len(row_data) > 16 else "",
                "Setting": row_data[17] if len(row_data) > 17 else "",
                "Git Branch": row_data[18] if len(row_data) > 18 else "",
                "Git Commit": row_data[19] if len(row_data) > 19 else "",
            }

            # Add dataset-specific columns if they don't exist
            metric_names = ["Train Loss", "Val Loss", "Test Loss", "Test RMSE", "Test MSE", "Test MAE", "Test MAPE", "Test MSPE"]
            dataset_metrics = {}

            for i, metric in enumerate(metric_names):
                col_name = f"{dataset}_{metric.replace(' ', '_')}"
                if i + 3 < len(row_data):  # Train Loss starts at index 3
                    dataset_metrics[col_name] = row_data[i + 3]

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
            if new_headers != headers:
                lb_all_worksheet.update("1:1", [new_headers])
                headers = new_headers

                # Apply colors to dataset columns
                self._apply_dataset_colors(lb_all_worksheet, headers)

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
                # Update existing row
                cell_range = f"A{trial_row_idx}:{chr(ord('A') + len(headers) - 1)}{trial_row_idx}"
                lb_all_worksheet.update(cell_range, [updated_row])
            else:
                # Add new row
                lb_all_worksheet.append_row(updated_row)

            print(f"LB_ALL updated for trial: {trial_name}, dataset: {dataset}")

            # Apply best value highlighting after update
            self._highlight_best_values_in_lb_all(lb_all_worksheet)

        except Exception as e:
            print(f"Failed to update LB_ALL trial {trial_name}: {e}")

    def _update_lb_all_view(self) -> None:
        """Legacy function - now handled per-trial"""
        pass

    def upload_results(self, args: Any, train_loss: float, val_loss: float, test_loss: float, test_metrics: dict[str, float], setting: str) -> None:
        """Upload experiment results to Google Sheets"""

        if not GSPREAD_AVAILABLE or not self.worksheet:
            print("Google Sheets upload skipped (not available)")
            self._save_to_local_backup(args, train_loss, val_loss, test_loss, test_metrics, setting)
            return

        try:
            # Setup headers if needed
            self._setup_headers()

            # Get git information
            git_branch, git_commit = get_git_info()

            # Prepare row data exactly as requested
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            trial_name = getattr(args, "trial_name", "") or f"Trial_{timestamp.replace(' ', '_').replace(':', '')}"
            dataset = getattr(args, "data", "Unknown")  # ETTm2, ETTh1, etc.

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

            # Add row to main LB sheet
            self.worksheet.append_row(row_data)

            # Add to dataset-specific worksheet
            dataset_worksheet = self._setup_dataset_worksheet(dataset)
            if dataset_worksheet:
                # Prepare row data without Dataset column
                dataset_row_data = [
                    trial_name,
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
                dataset_worksheet.append_row(dataset_row_data)

                # Apply best value highlighting to dataset worksheet
                self._highlight_best_values_in_dataset_worksheet(dataset_worksheet)

            # Apply formatting to highlight best results
            self._highlight_best_results()

            # Update LB_ALL view with this trial's data
            self._update_lb_all_trial(trial_name, dataset, row_data)

            print(f"Results uploaded to Google Sheets successfully for trial: {trial_name}")
            print(f"  Dataset: {dataset}")
            print(f"  Git: {git_branch}@{git_commit}")
            print(f"  Test RMSE: {test_metrics.get('rmse', 0):.8f}")
            print(f"  Test MSE: {test_metrics.get('mse', 0):.8f}")
            print(f"  Test MAE: {test_metrics.get('mae', 0):.8f}")

        except Exception as e:
            print(f"Failed to upload results to Google Sheets: {e}")
            # Save to local backup as fallback
            self._save_to_local_backup(args, train_loss, val_loss, test_loss, test_metrics, setting)

    def _highlight_best_results(self) -> None:
        """Highlight best results (lowest for loss metrics) with improved precision"""
        if not self.worksheet:
            return

        try:
            # Get all data
            all_data = self.worksheet.get_all_values()
            if len(all_data) <= 1:  # Only headers
                return

            headers = all_data[0]
            data_rows = all_data[1:]

            # Define columns to highlight (all should be minimized for time series forecasting)
            metric_columns = ["Train Loss", "Val Loss", "Test Loss", "Test RMSE", "Test MSE", "Test MAE", "Test MAPE", "Test MSPE"]

            # Group by dataset for comparison
            dataset_groups: dict[str, list[tuple[int, list[str]]]] = {}
            dataset_col_idx = headers.index("Dataset") if "Dataset" in headers else None

            if dataset_col_idx is not None:
                for i, row in enumerate(data_rows):
                    if dataset_col_idx < len(row):
                        dataset = row[dataset_col_idx]
                        if dataset not in dataset_groups:
                            dataset_groups[dataset] = []
                        dataset_groups[dataset].append((i, row))

            # Highlight best results for each dataset separately
            for dataset, rows in dataset_groups.items():
                for metric_name in metric_columns:
                    if metric_name not in headers:
                        continue

                    col_idx = headers.index(metric_name)

                    # Get values for this column within this dataset
                    values = []
                    row_indices = []
                    for row_idx, row in rows:
                        try:
                            val = float(row[col_idx]) if col_idx < len(row) and row[col_idx] else float("inf")
                            # Round to 5 decimal places for comparison
                            val = round(val, 5)
                            values.append(val)
                            row_indices.append(row_idx)
                        except (ValueError, IndexError):
                            values.append(float("inf"))
                            row_indices.append(row_idx)

                    if not values:
                        continue

                    # Find best value (minimum for all metrics) with 5 decimal precision
                    valid_values = [val for val in values if val != float("inf")]
                    if not valid_values:
                        continue

                    best_val = min(valid_values)

                    # Find rows with best value and format them
                    for i, val in enumerate(values):
                        if abs(val - best_val) < 1e-6 and val != float("inf"):  # Use small epsilon for float comparison
                            row_num = row_indices[i] + 2  # +2 because of header and 0-indexed
                            col_letter = chr(ord("A") + col_idx)
                            cell_range = f"{col_letter}{row_num}"
                            self.worksheet.format(cell_range, {"textFormat": {"bold": True}, "backgroundColor": {"red": 0.9, "green": 1.0, "blue": 0.9}})

        except Exception as e:
            print(f"Failed to highlight best results: {e}")

    def _save_to_local_backup(self, args: Any, train_loss: float, val_loss: float, test_loss: float, test_metrics: dict[str, float], setting: str) -> None:
        """Save results to local JSON file as backup"""
        try:
            # Get git information
            git_branch, git_commit = get_git_info()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            trial_name = getattr(args, "trial_name", "") or f"Trial_{timestamp.replace(' ', '_').replace(':', '')}"
            dataset = getattr(args, "data", "Unknown")

            data_row = {
                "timestamp": timestamp,
                "trial_name": trial_name,
                "dataset": dataset,
                "model": getattr(args, "model", ""),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "test_rmse": test_metrics.get("rmse", 0),
                "test_mse": test_metrics.get("mse", 0),
                "test_mae": test_metrics.get("mae", 0),
                "test_mape": test_metrics.get("mape", 0),
                "test_mspe": test_metrics.get("mspe", 0),
                "seq_len": getattr(args, "seq_len", ""),
                "pred_len": getattr(args, "pred_len", ""),
                "batch_size": getattr(args, "batch_size", ""),
                "learning_rate": getattr(args, "learning_rate", ""),
                "description": getattr(args, "des", ""),
                "setting": setting,
                "git_branch": git_branch,
                "git_commit": git_commit,
            }

            backup_file = "experiment_results_backup.json"

            # Load existing data
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, encoding="utf-8") as f:
                        data = json.load(f)
                except:
                    data = []
            else:
                data = []

            # Add new row
            data.append(data_row)

            # Save back to file
            with open(backup_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"Results saved to local backup: {backup_file}")

        except Exception as e:
            print(f"Failed to save local backup: {e}")

    def get_leaderboard(self, dataset_name: str | None = None) -> Any:
        """Get current leaderboard data"""
        if not self.worksheet:
            return []

        try:
            all_data = self.worksheet.get_all_records()

            if dataset_name:
                # Filter by dataset
                filtered_data = [row for row in all_data if row.get("Dataset", "").lower() == dataset_name.lower()]
                return filtered_data

            return all_data

        except Exception as e:
            print(f"Failed to get leaderboard data: {e}")
            return []


# Global instance
_sheets_uploader = None


def get_sheets_uploader() -> SheetsUploader | None:
    """Get global sheets uploader instance"""
    global _sheets_uploader

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

        _sheets_uploader = SheetsUploader(spreadsheet_url, credentials_path)

    return _sheets_uploader


def upload_experiment_results(args: Any, train_loss: float, val_loss: float, test_loss: float, test_metrics: dict[str, float], setting: str) -> None:
    """Convenience function to upload experiment results"""
    uploader = get_sheets_uploader()
    if uploader:
        uploader.upload_results(args, train_loss, val_loss, test_loss, test_metrics, setting)
