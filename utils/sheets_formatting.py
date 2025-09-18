"""
Google Sheets formatting and highlighting module
Handles color coding for datasets and best value highlighting.
"""

from typing import Any


class SheetsFormatter:
    """Class to handle Google Sheets formatting operations"""

    def __init__(self) -> None:
        self.dataset_colors = [
            {"red": 0.95, "green": 0.85, "blue": 0.85},  # Light red
            {"red": 0.85, "green": 0.95, "blue": 0.85},  # Light green
            {"red": 0.85, "green": 0.85, "blue": 0.95},  # Light blue
            {"red": 0.95, "green": 0.95, "blue": 0.85},  # Light yellow
            {"red": 0.95, "green": 0.85, "blue": 0.95},  # Light magenta
            {"red": 0.85, "green": 0.95, "blue": 0.95},  # Light cyan
            {"red": 0.92, "green": 0.88, "blue": 0.85},  # Light peach
            {"red": 0.88, "green": 0.85, "blue": 0.92},  # Light lavender
        ]

    def get_dataset_color(self, dataset_index: int) -> dict[str, dict[str, float]]:
        """Get background color for dataset columns"""
        return {"backgroundColor": self.dataset_colors[dataset_index % len(self.dataset_colors)]}

    def apply_dataset_colors(self, worksheet: Any, headers: list[str]) -> None:
        """Apply colors to dataset-specific columns in LB_ALL worksheet"""
        try:
            # Identify base headers (settings columns) that should have gray background
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

            # Apply light gray background to base/settings columns
            settings_background = {"red": 0.93, "green": 0.93, "blue": 0.93}  # Light gray
            for i, header in enumerate(headers):
                if header in base_headers:
                    col_letter = self._get_column_letter(i)
                    # Format header
                    header_format = {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": settings_background}
                    worksheet.format(f"{col_letter}1", header_format)

                    # Format data rows with same light background
                    range_str = f"{col_letter}2:{col_letter}"
                    worksheet.format(range_str, {"backgroundColor": settings_background, "textFormat": {"fontSize": 10}})
                    print(f"Applied settings background to column {i + 1}: {header}")

            # Group dataset columns by dataset name
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
                color_format = self.get_dataset_color(dataset_index)
                print(f"Applying color {dataset_index} to dataset {dataset_name} columns: {column_indices}")

                for col_idx in column_indices:
                    # Format entire column (header + data)
                    col_letter = self._get_column_letter(col_idx)
                    # Format header specifically with bold and dataset color
                    header_format = {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": color_format["backgroundColor"]}
                    worksheet.format(f"{col_letter}1", header_format)

                    # Format data rows with slightly lighter color
                    data_color = color_format["backgroundColor"].copy()
                    # Make data rows slightly lighter
                    for color_key in data_color:
                        data_color[color_key] = min(1.0, data_color[color_key] + 0.03)

                    # Get the column range for data rows
                    range_str = f"{col_letter}2:{col_letter}"
                    # Apply background color and font size
                    worksheet.format(range_str, {"backgroundColor": data_color, "textFormat": {"fontSize": 10}})

                dataset_index += 1

            print(f"Applied colors to {len(dataset_columns)} datasets in LB_ALL worksheet")

        except Exception as e:
            print(f"Failed to apply dataset colors: {e}")
            import traceback

            traceback.print_exc()

    def apply_dataset_colors_selective(self, worksheet: Any, headers: list[str], target_columns: list[str]) -> None:
        """Apply dataset colors only to specific columns to preserve existing formatting"""
        try:
            if not headers or not target_columns:
                print("No headers or target columns provided for selective color application")
                return

            # Group target columns by dataset
            dataset_columns = {}
            for col_name in target_columns:
                if col_name in headers:
                    # Extract dataset name from column name
                    dataset_name = col_name.split("_")[0] if "_" in col_name else col_name
                    if dataset_name not in dataset_columns:
                        dataset_columns[dataset_name] = []
                    col_idx = headers.index(col_name)
                    dataset_columns[dataset_name].append(col_idx)

            if not dataset_columns:
                print("No valid dataset columns found in target columns")
                return

            print(f"Applying selective colors to {len(dataset_columns)} datasets: {list(dataset_columns.keys())}")

            # Apply colors to each dataset's columns
            dataset_index = 0
            for dataset_name, column_indices in sorted(dataset_columns.items()):
                color_format = self.get_dataset_color(dataset_index)
                print(f"Applying color {dataset_index} to dataset {dataset_name} columns: {column_indices}")

                for col_idx in column_indices:
                    # Format entire column (header + data)
                    col_letter = self._get_column_letter(col_idx)
                    # Format header specifically with bold and dataset color
                    header_format = {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": color_format["backgroundColor"]}
                    worksheet.format(f"{col_letter}1", header_format)

                    # Format data rows with slightly lighter color
                    data_color = color_format["backgroundColor"].copy()
                    # Make data rows slightly lighter
                    for color_key in data_color:
                        data_color[color_key] = min(1.0, data_color[color_key] + 0.03)

                    # Get the column range for data rows
                    range_str = f"{col_letter}2:{col_letter}"
                    # Apply background color and font size only, preserve bold formatting
                    worksheet.format(range_str, {"backgroundColor": data_color, "textFormat": {"fontSize": 10}})

                dataset_index += 1

            print(f"Applied selective colors to {len(target_columns)} columns")

        except Exception as e:
            print(f"Failed to apply selective dataset colors: {e}")
            import traceback

            traceback.print_exc()

    def highlight_best_values_in_lb_all(self, worksheet: Any, specific_columns: list[str] = None) -> None:
        """Highlight best values (lowest) within each metric column with bold text

        Args:
            worksheet: The Google Sheets worksheet
            specific_columns: If provided, only highlight these specific columns instead of all columns
        """
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

            # Determine which columns to process
            if specific_columns:
                # Only process the specified columns that exist in headers
                columns_to_process = [col for col in specific_columns if col in headers]
                print(f"Processing only specific columns: {columns_to_process}")
            else:
                # Process all metric columns (existing behavior)
                columns_to_process = [header for header in headers if header not in base_headers and "_" in header]
                print(f"Processing all metric columns: {len(columns_to_process)} columns")

            # Process each specified column
            processed_columns = 0

            for header in columns_to_process:
                col_idx = headers.index(header)

                # Extract values from this column (only non-empty values)
                values = []
                row_indices = []

                for row_idx, row in enumerate(data_rows):
                    if col_idx < len(row) and row[col_idx] and row[col_idx].strip():
                        try:
                            # Parse the value
                            val = float(row[col_idx])
                            values.append(val)
                            row_indices.append(row_idx)
                        except (ValueError, TypeError):
                            continue

                # Skip if no valid values
                if not values:
                    print(f"No valid values found for column {col_idx}: {header}")
                    continue

                # Find minimum value (best for all metrics)
                min_val = min(values)
                print(f"Processing column {col_idx}: {header} - min value: {min_val} from {len(values)} values")

                # Apply bold formatting to cells with best value
                col_letter = self._get_column_letter(col_idx)

                # Batch format operations for efficiency
                try:
                    # First, remove bold from all cells in this column (batch operation)
                    column_range = f"{col_letter}2:{col_letter}{len(data_rows) + 1}"
                    worksheet.format(column_range, {"textFormat": {"bold": False, "fontSize": 10}})

                    # Then apply bold to cells with minimum value
                    bold_applied = 0
                    for i, val in enumerate(values):
                        if abs(val - min_val) < 1e-10:  # Use small epsilon for float comparison
                            row_num = row_indices[i] + 2  # +2 because header is row 1
                            cell_ref = f"{col_letter}{row_num}"
                            try:
                                worksheet.format(cell_ref, {"textFormat": {"bold": True, "fontSize": 10}})
                                bold_applied += 1
                            except Exception as e:
                                print(f"Failed to apply bold to {cell_ref}: {e}")

                except Exception as e:
                    print(f"Failed to format column {header}: {e}")
                    # Fallback to individual cell formatting
                    for i in range(len(data_rows)):
                        cell_ref = f"{col_letter}{i + 2}"
                        try:
                            worksheet.format(cell_ref, {"textFormat": {"bold": False, "fontSize": 10}})
                        except Exception:
                            pass

                    bold_applied = 0
                    for i, val in enumerate(values):
                        if abs(val - min_val) < 1e-10:
                            row_num = row_indices[i] + 2
                            cell_ref = f"{col_letter}{row_num}"
                            try:
                                worksheet.format(cell_ref, {"textFormat": {"bold": True, "fontSize": 10}})
                                bold_applied += 1
                            except Exception:
                                pass

                print(f"Applied bold to {bold_applied} cells in column {header}")
                processed_columns += 1

            print(f"Best values highlighted in LB_ALL worksheet ({processed_columns} columns processed)")

        except Exception as e:
            print(f"Failed to highlight best values in LB_ALL: {e}")

    def highlight_best_values_in_dataset(self, worksheet: Any) -> None:
        """Highlight best values (lowest) within each metric column in dataset worksheet"""
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

            # Define metric columns to highlight dynamically based on what exists
            possible_metric_columns = [
                # Long-term forecast metrics
                "Train Loss",
                "Val Loss",
                "Test Loss",
                "Test RMSE",
                "Test MSE",
                "Test MAE",
                "Test MAPE",
                "Test MSPE",
                # Classification metrics
                "Accuracy",
                # Short-term forecast metrics
                "SMAPE",
                "MASE",
                "OWA",
            ]

            # Also check for dataset-specific and seasonal pattern columns
            additional_metric_patterns = ["_Acc", "_SMAPE", "_MASE", "_OWA"]

            # Find actual metric columns present in this worksheet
            metric_columns = []
            for header in headers:
                if header in possible_metric_columns:
                    metric_columns.append(header)
                elif any(pattern in header for pattern in additional_metric_patterns):
                    metric_columns.append(header)

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
                col_letter = self._get_column_letter(col_idx)

                # First, remove bold from all cells in this column
                for i in range(len(data_rows)):
                    cell_ref = f"{col_letter}{i + 2}"  # +2 because row 1 is header
                    try:
                        worksheet.format(cell_ref, {"textFormat": {"bold": False, "fontSize": 10}})
                    except Exception:
                        pass

                # Then apply bold to cells with minimum value
                for i, val in enumerate(values):
                    if abs(val - min_val) < 1e-10:  # Use small epsilon for float comparison
                        row_num = row_indices[i] + 2  # +2 because header is row 1
                        cell_ref = f"{col_letter}{row_num}"
                        worksheet.format(cell_ref, {"textFormat": {"bold": True, "fontSize": 10}})

            print("Best values highlighted in dataset worksheet")

        except Exception as e:
            print(f"Failed to highlight best values in dataset worksheet: {e}")

    def apply_dataset_formatting(self, worksheet: Any) -> None:
        """Apply formatting to dataset-specific worksheets with gray settings and colored results"""
        try:
            # Get headers
            all_data = worksheet.get_all_values()
            if not all_data:
                return
            headers = all_data[0]

            # Define settings columns (gray) - common across all tasks
            settings_columns = [
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

            # Determine results columns based on what's present in headers
            # This makes it dynamic for different task types
            possible_results_columns = [
                # Long-term forecast metrics
                "Train Loss",
                "Val Loss",
                "Test Loss",
                "Test RMSE",
                "Test MSE",
                "Test MAE",
                "Test MAPE",
                "Test MSPE",
                # Classification metrics
                "Accuracy",
                # Short-term forecast metrics
                "SMAPE",
                "MASE",
                "OWA",
            ]

            # Also check for dataset-specific accuracy columns (e.g., EthanolConcentration_Acc)
            # and seasonal pattern columns (e.g., Monthly_SMAPE, Yearly_MASE)
            additional_results_patterns = ["_Acc", "_SMAPE", "_MASE", "_OWA"]

            # Find actual results columns present in this worksheet
            results_columns = []
            for header in headers:
                if header in possible_results_columns:
                    results_columns.append(header)
                elif any(pattern in header for pattern in additional_results_patterns):
                    results_columns.append(header)

            # Apply gray background to settings columns
            settings_background = {"red": 0.93, "green": 0.93, "blue": 0.93}  # Light gray
            for i, header in enumerate(headers):
                if header in settings_columns:
                    col_letter = self._get_column_letter(i)
                    # Format header
                    header_format = {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": settings_background}
                    worksheet.format(f"{col_letter}1", header_format)

                    # Format data rows with same gray background
                    range_str = f"{col_letter}2:{col_letter}"
                    worksheet.format(range_str, {"backgroundColor": settings_background, "textFormat": {"fontSize": 10}})

            # Apply colored background to results columns
            results_background = {"red": 0.85, "green": 0.95, "blue": 0.85}  # Light green for results
            for i, header in enumerate(headers):
                if header in results_columns:
                    col_letter = self._get_column_letter(i)
                    # Format header with color
                    header_format = {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": results_background}
                    worksheet.format(f"{col_letter}1", header_format)

                    # Format data rows with lighter version of the color
                    data_color = results_background.copy()
                    # Make data rows slightly lighter
                    for color_key in data_color:
                        data_color[color_key] = min(1.0, data_color[color_key] + 0.03)

                    range_str = f"{col_letter}2:{col_letter}"
                    worksheet.format(range_str, {"backgroundColor": data_color, "textFormat": {"fontSize": 10}})

            # Apply best value highlighting to results columns only
            self.highlight_best_values_in_dataset(worksheet)

            print("Applied dataset-specific formatting with separated settings and results")

        except Exception as e:
            print(f"Failed to apply dataset formatting: {e}")
            import traceback

            traceback.print_exc()

    def highlight_best_results_by_dataset(self, worksheet: Any) -> None:
        # Skip formatting if disabled via environment variable
        import os

        if os.getenv("SHEETS_ENABLE_FORMATTING", "true").lower() != "true":
            print("Formatting disabled via SHEETS_ENABLE_FORMATTING environment variable")
            return
        """Highlight best results (lowest for loss metrics) in main LB worksheet by dataset"""
        try:
            # Get all data
            all_data = worksheet.get_all_values()
            if len(all_data) <= 1:  # Only headers
                return

            headers = all_data[0]
            data_rows = all_data[1:]

            # Define columns to highlight dynamically based on what exists in headers
            possible_metric_columns = [
                # Long-term forecast metrics
                "Train Loss",
                "Val Loss",
                "Test Loss",
                "Test RMSE",
                "Test MSE",
                "Test MAE",
                "Test MAPE",
                "Test MSPE",
                # Classification metrics
                "Accuracy",
                # Short-term forecast metrics
                "SMAPE",
                "MASE",
                "OWA",
            ]

            # Also check for dataset-specific and seasonal pattern columns
            additional_metric_patterns = ["_Acc", "_SMAPE", "_MASE", "_OWA"]

            # Find actual metric columns present in this worksheet
            metric_columns = []
            for header in headers:
                if header in possible_metric_columns:
                    metric_columns.append(header)
                elif any(pattern in header for pattern in additional_metric_patterns):
                    metric_columns.append(header)

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
            for _, rows in dataset_groups.items():
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

                    # Collect all cells to format for batch processing
                    cells_to_format = []
                    for i, val in enumerate(values):
                        if abs(val - best_val) < 1e-6 and val != float("inf"):  # Use small epsilon for float comparison
                            row_num = row_indices[i] + 2  # +2 because of header and 0-indexed
                            col_letter = self._get_column_letter(col_idx)
                            cell_range = f"{col_letter}{row_num}"
                            cells_to_format.append(cell_range)

                    # Batch format all cells at once to reduce API calls
                    if cells_to_format:
                        try:
                            # Create batch format request
                            batch_formats = []
                            for cell_range in cells_to_format:
                                batch_formats.append(
                                    {
                                        "range": cell_range,
                                        "format": {"textFormat": {"bold": True, "fontSize": 10}, "backgroundColor": {"red": 0.9, "green": 1.0, "blue": 0.9}},
                                    }
                                )

                            # Apply all formats in a single API call
                            worksheet.batch_format(batch_formats)
                            print(f"Applied bold to {len(cells_to_format)} cells in column {headers[col_idx] if col_idx < len(headers) else 'Unknown'}")

                            # Add small delay to respect rate limits
                            import time

                            time.sleep(0.1)

                        except Exception as format_error:
                            if "429" in str(format_error) or "Quota exceeded" in str(format_error):
                                print(
                                    f"API quota exceeded for column {headers[col_idx] if col_idx < len(headers) else 'Unknown'}. Skipping formatting for this column."
                                )
                            else:
                                print(f"Failed to format column {headers[col_idx] if col_idx < len(headers) else 'Unknown'}: {format_error}")
                            # Continue without formatting if quota exceeded
                            continue

        except Exception as e:
            print(f"Failed to highlight best results: {e}")

    def _get_column_letter(self, col_idx: int) -> str:
        """Convert column index to Excel-style column letter (A, B, ..., Z, AA, AB, ...)"""
        result = ""
        while col_idx >= 0:
            result = chr(ord("A") + (col_idx % 26)) + result
            col_idx = col_idx // 26 - 1
            if col_idx < 0:
                break
        return result
