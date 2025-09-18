"""
Base Google Sheets operations module
Handles basic connection, worksheet setup, and data upload operations.
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


class BaseSheetsHandler:
    """Base class for Google Sheets operations"""

    def __init__(self, spreadsheet_url: str, credentials_path: str | None = None):
        self.spreadsheet_url = spreadsheet_url
        self.credentials_path = credentials_path
        self.gc: Any = None
        self.sheet: Any = None
        self.main_worksheet: Any = None

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
                self.main_worksheet = self.sheet.worksheet("LB")
            except:
                self.main_worksheet = self.sheet.add_worksheet(title="LB", rows=1000, cols=25)

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

    def _setup_main_headers(self) -> None:
        """Setup column headers for main LB worksheet"""
        if not self.main_worksheet:
            return

        try:
            # Check if headers already exist
            existing_headers = self.main_worksheet.row_values(1)
            if existing_headers and len(existing_headers) > 1:
                return

            # Define headers for main LB worksheet
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
            self.main_worksheet.update("1:1", [headers])

            # Format headers (bold with background color)
            self.main_worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})

            print("Main headers setup completed")

        except Exception as e:
            print(f"Failed to setup main headers: {e}")

    def get_or_create_worksheet(self, name: str, rows: int = 1000, cols: int = 25) -> Any:
        """Get existing worksheet or create new one"""
        if not self.gc or not self.sheet:
            return None

        try:
            # Try to get existing worksheet
            try:
                worksheet = self.sheet.worksheet(name)
            except:
                # Create new worksheet
                worksheet = self.sheet.add_worksheet(title=name, rows=rows, cols=cols)

            return worksheet

        except Exception as e:
            print(f"Failed to get/create worksheet {name}: {e}")
            return None

    def setup_dataset_worksheet(self, worksheet_name: str) -> Any:
        """Setup dataset-specific worksheet"""
        worksheet = self.get_or_create_worksheet(worksheet_name, 1000, 25)
        if not worksheet:
            return None

        try:
            # Setup headers for dataset worksheet with settings first, then results
            headers = [
                # Settings columns (gray background)
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
                # Results columns (colored background)
                "Train Loss",
                "Val Loss",
                "Test Loss",
                "Test RMSE",
                "Test MSE",
                "Test MAE",
                "Test MAPE",
                "Test MSPE",
            ]

            # Check if headers already exist
            existing_headers = worksheet.row_values(1)
            if not existing_headers or len(existing_headers) <= 1:
                # Set headers
                worksheet.update("1:1", [headers])
                # Format headers
                worksheet.format("1:1", {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93}})

            return worksheet

        except Exception as e:
            print(f"Failed to setup dataset worksheet {worksheet_name}: {e}")
            return None

    def setup_lb_all_worksheet(self) -> Any:
        """Setup LB_ALL worksheet with trial-centric view"""
        worksheet = self.get_or_create_worksheet("LB_ALL", 1000, 50)

        # Ensure basic structure exists (headers will be added by specific functions as needed)
        if worksheet:
            try:
                existing_data = worksheet.get_all_values()
                if not existing_data:
                    print("LB_ALL worksheet exists but is empty - headers will be added by specific upload functions")
            except Exception as e:
                print(f"Warning: Could not check LB_ALL worksheet content: {e}")

        return worksheet

    def add_row_to_worksheet(self, worksheet: Any, row_data: list[str]) -> bool:
        """Add a row to specified worksheet"""
        try:
            if worksheet:
                print(f"DEBUG: Attempting to add row with {len(row_data)} columns")
                print(f"DEBUG: Worksheet name: {getattr(worksheet, 'title', 'Unknown')}")
                print(f"DEBUG: Row data preview: {row_data[:5]}...")
                worksheet.append_row(row_data)
                print("DEBUG: Row successfully appended to worksheet")
                return True
            else:
                print("DEBUG: Worksheet is None, cannot add row")
                return False
        except Exception as e:
            print(f"Failed to add row to worksheet: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            import traceback

            traceback.print_exc()
            return False

    def save_to_local_backup(self, args: Any, train_loss: float, val_loss: float, test_loss: float, test_metrics: dict[str, float], setting: str) -> None:
        """Save results to local JSON file as backup"""
        try:
            # Get git information
            git_branch, git_commit = get_git_info()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            trial_name = getattr(args, "trial_name", "") or f"Trial_{timestamp.replace(' ', '_').replace(':', '')}"
            # Use model_id for dataset name (e.g., ECL_96_96) instead of data parameter (custom)
            dataset = getattr(args, "model_id", getattr(args, "data", "Unknown"))

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
        if not self.main_worksheet:
            return []

        try:
            all_data = self.main_worksheet.get_all_records()

            if dataset_name:
                # Filter by dataset
                filtered_data = [row for row in all_data if row.get("Dataset", "").lower() == dataset_name.lower()]
                return filtered_data

            return all_data

        except Exception as e:
            print(f"Failed to get leaderboard data: {e}")
            return []
