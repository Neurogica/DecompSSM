# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
M4 Dataset
"""

import logging
import os
import pathlib
import sys
from dataclasses import dataclass
from urllib import request

import numpy as np
import pandas as pd


def url_file_name(url: str) -> str:
    """
    Extract file name from url.

    :param url: URL to extract file name from.
    :return: File name.
    """
    return url.split("/")[-1] if len(url) > 0 else ""


def download(url: str, file_path: str) -> None:
    """
    Download a file to the given path.

    :param url: URL to download
    :param file_path: Where to download the content.
    """

    def progress(count, block_size, total_size):
        progress_pct = float(count * block_size) / float(total_size) * 100.0
        sys.stdout.write(f"\rDownloading {url} to {file_path} {progress_pct:.1f}%")
        sys.stdout.flush()

    if not os.path.isfile(file_path):
        opener = request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        request.install_opener(opener)
        pathlib.Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        f, _ = request.urlretrieve(url, file_path, progress)
        sys.stdout.write("\n")
        sys.stdout.flush()
        file_info = os.stat(f)
        logging.info(f"Successfully downloaded {os.path.basename(file_path)} {file_info.st_size} bytes.")
    else:
        file_info = os.stat(file_path)
        logging.info(f"File already exists: {file_path} {file_info.st_size} bytes.")


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def initialize(dataset_file: str = "../dataset/m4") -> None:
        """
        Initialize M4 dataset by converting CSV files to NPZ format.
        This should be called once to prepare the dataset.
        """
        print("Initializing M4 dataset...")

        train_cache_file = os.path.join(dataset_file, "training.npz")
        test_cache_file = os.path.join(dataset_file, "test.npz")

        # Skip if files already exist
        if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
            print("M4 dataset already initialized.")
            return

        # Define CSV file paths
        train_files = [
            os.path.join(dataset_file, "Train", "Yearly-train.csv"),
            os.path.join(dataset_file, "Train", "Quarterly-train.csv"),
            os.path.join(dataset_file, "Train", "Monthly-train.csv"),
            os.path.join(dataset_file, "Train", "Weekly-train.csv"),
            os.path.join(dataset_file, "Train", "Daily-train.csv"),
            os.path.join(dataset_file, "Train", "Hourly-train.csv"),
        ]

        test_files = [
            os.path.join(dataset_file, "Test", "Yearly-test.csv"),
            os.path.join(dataset_file, "Test", "Quarterly-test.csv"),
            os.path.join(dataset_file, "Test", "Monthly-test.csv"),
            os.path.join(dataset_file, "Test", "Weekly-test.csv"),
            os.path.join(dataset_file, "Test", "Daily-test.csv"),
            os.path.join(dataset_file, "Test", "Hourly-test.csv"),
        ]

        # Process training data
        print("Processing training data...")
        train_data = []
        for file_path in train_files:
            if os.path.exists(file_path):
                print(f"Processing {file_path}")
                df = pd.read_csv(file_path, header=0)
                # Skip the first column (ID column) and convert to numpy array
                values = df.values[:, 1:].astype(float)
                train_data.append(values)
            else:
                print(f"Warning: {file_path} not found")

        if train_data:
            # Combine all series from different frequencies into one list
            all_train_series = []
            for data_matrix in train_data:
                for series in data_matrix:
                    all_train_series.append(series[~np.isnan(series)])  # Remove NaN values

            print(f"Saving training data to {train_cache_file}")
            np.savez_compressed(train_cache_file, data=np.array(all_train_series, dtype=object))
            print(f"Training data: {len(all_train_series)} time series")

        # Process test data
        print("Processing test data...")
        test_data = []
        for file_path in test_files:
            if os.path.exists(file_path):
                print(f"Processing {file_path}")
                df = pd.read_csv(file_path, header=0)
                # Skip the first column (ID column) and convert to numpy array
                values = df.values[:, 1:].astype(float)
                test_data.append(values)
            else:
                print(f"Warning: {file_path} not found")

        if test_data:
            # Combine all series from different frequencies into one list
            all_test_series = []
            for data_matrix in test_data:
                for series in data_matrix:
                    all_test_series.append(series[~np.isnan(series)])  # Remove NaN values

            print(f"Saving test data to {test_cache_file}")
            np.savez_compressed(test_cache_file, data=np.array(all_test_series, dtype=object))
            print(f"Test data: {len(all_test_series)} time series")

        print("M4 dataset initialization completed!")

    @staticmethod
    def load(training: bool = True, dataset_file: str = "../dataset/m4") -> "M4Dataset":
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        info_file = os.path.join(dataset_file, "M4-info.csv")
        train_cache_file = os.path.join(dataset_file, "training.npz")
        test_cache_file = os.path.join(dataset_file, "test.npz")
        m4_info = pd.read_csv(info_file)
        npz_data = np.load(train_cache_file if training else test_cache_file, allow_pickle=True)
        return M4Dataset(
            ids=m4_info.M4id.values,
            groups=m4_info.SP.values,
            frequencies=m4_info.Frequency.values,
            horizons=m4_info.Horizon.values,
            values=npz_data["data"],
        )


@dataclass()
class M4Meta:
    seasonal_patterns = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {"Yearly": 6, "Quarterly": 8, "Monthly": 18, "Weekly": 13, "Daily": 14, "Hourly": 48}  # different predict length
    frequency_map = {"Yearly": 1, "Quarterly": 4, "Monthly": 12, "Weekly": 1, "Daily": 1, "Hourly": 24}
    history_size = {"Yearly": 1.5, "Quarterly": 1.5, "Monthly": 1.5, "Weekly": 10, "Daily": 10, "Hourly": 10}  # from interpretable.gin


def load_m4_info(info_file_path: str) -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(info_file_path)
