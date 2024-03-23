# Epileptic Dataset

import os
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from mne.time_frequency import psd_array_multitaper

import torch
from torch.utils.data import Dataset
from model_arguments import ModelArguments


class EpilepsyDataset(Dataset):
    """
    Dataset class for handling epileptic data.
    """

    def __init__(
        self, 
        args: ModelArguments
    ):
        """
        Initializes the EpilepsyDataset with the given parameters.

        Parameters:
        - path_to_annotation_file (str): Path to the annotation file.
        - path_to_data (str): Path to the data directory.
        - signals_names (list[str]): List of signal names.
        - signal_length (int): Length of a signal.
        """
        super().__init__()
        self._path_to_annotation_file = args.path_to_annotation
        self._path_to_data = args.path_to_data
        self._signals_names = args.signals_names
        self._number_of_features = args.number_of_channels
        
        self._annotation_file = pd.read_csv(self._path_to_annotation_file)

        # Class balancing
        class_0 = self._annotation_file[self._annotation_file["Label"] == 0]
        class_1 = self._annotation_file[self._annotation_file["Label"] == 1]

        num_samples_to_keep = min(len(class_0), len(class_1))
        self._annotation_file = pd.concat(
            [class_0.sample(num_samples_to_keep, random_state=42), class_1], axis=0
        )

    @staticmethod
    def _normalize_list_of_arrays(list_of_arrays: list[np.array]) -> list[np.array]:
        """
        Normalize a list of arrays.

        Parameters:
        - list_of_arrays (list[np.array]): List of arrays to normalize.

        Returns:
        - list[np.array]: List of normalized arrays.
        """

        def normalize_array(array):
            array_min = np.min(array)
            array_max = np.max(array)
            normalized_array = (array - array_min) / (array_max - array_min)
            return normalized_array

        normalized_list = [normalize_array(array) for array in list_of_arrays]
        return normalized_list

    @staticmethod
    def _get_acc_sqi(
        acc_x: np.array, acc_y: np.array, acc_z: np.array, sampling_rate: int = 128
    ) -> float:
        """
        Calculate the acceleration-based signal quality index (SQI).

        Parameters:
        - acc_x (np.array): Acceleration data along the X-axis.
        - acc_y (np.array): Acceleration data along the Y-axis.
        - acc_z (np.array): Acceleration data along the Z-axis.
        - sampling_rate (int): Sampling rate of the accelerometer data.

        Returns:
        - float: Signal quality index.
        """
        acc_data = np.sqrt((acc_x * acc_x + acc_y * acc_y + acc_z * acc_z) / 3.0)

        segment_length = 4
        segment_samples = segment_length * sampling_rate
        num_segments = len(acc_data) // segment_samples

        narrowband_powers = []
        broadband_powers = []

        for i in range(num_segments):
            segment_start = i * segment_samples
            segment_end = (i + 1) * segment_samples

            segment_data = acc_data[segment_start:segment_end]

            f_acc, pxx_acc = periodogram(segment_data, fs=sampling_rate)

            idx_08hz = np.argmax(f_acc >= 0.8)
            idx_5hz = np.argmax(f_acc >= 5.0)

            narrowband_power = np.mean(pxx_acc[idx_08hz : idx_5hz + 1])
            broadband_power = np.mean(pxx_acc[idx_08hz:])

            narrowband_powers.append(narrowband_power)
            broadband_powers.append(broadband_power)

        avg_narrowband_power = np.mean(narrowband_powers)
        avg_broadband_power = np.mean(broadband_powers)

        signal_quality = avg_narrowband_power / avg_broadband_power

        return signal_quality

    @staticmethod
    def _get_signal_power(signal: np.array, sample_rate: int = 128) -> np.array:
        """
        Calculate the power spectrum of a signal.

        Parameters:
        - signal (np.array): Input signal.
        - sample_rate (int): Sampling rate of the signal.

        Returns:
        - np.array: Power spectrum of the signal.
        """
        power = np.power(np.abs(np.fft.fft(signal)), 2)
        return power.astype(float)

    @staticmethod
    def _get_spectral_features(
        freq_domain_data: np.array, sample_rate: int = 128
    ) -> tuple:
        """
        Calculate spectral features from frequency domain data.

        Parameters:
        - freq_domain_data (np.array): Data in the frequency domain.
        - sample_rate (int): Sampling rate of the data.

        Returns:
        - tuple: Mean and standard deviation of the data.
        """
        return freq_domain_data.mean(), freq_domain_data.std()

    @staticmethod
    def _get_time_features(time_domain_data: np.array, sample_rate: int = 128) -> tuple:
        """
        Calculate time-domain features.

        Parameters:
        - time_domain_data (np.array): Data in the time domain.
        - sample_rate (int): Sampling rate of the data.

        Returns:
        - tuple: Mean and standard deviation of the data.
        """
        return time_domain_data.mean(), time_domain_data.std()

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Number of samples.
        """
        return len(self._annotation_file)

    def __getitem__(self, idx: int) -> tuple:
        """
        Gets the data for a given index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - tuple: Concatenated array of signals and their spectra, and item's label.
        """
        data_row = self._annotation_file.iloc[idx]
        patient_name = data_row["Patient"]
        segment_id = str(data_row["Segment"])
        label = data_row["Label"]

        paths_to_segments = [
            "_".join(
                [os.path.join(self._path_to_data, patient_name), signal, segment_id]
            )
            + ".parquet"
            for signal in self._signals_names
        ]

        files = [pd.read_parquet(path) for path in paths_to_segments]
        signals = torch.tensor([x["data"] / float(10**9) for x in files])
        signal_length = len(signals[0])

        ms_in_hour = 3_600_000
        encoded_hours = torch.tensor(
            np.array([(((x / ms_in_hour) % 24) / 24) for x in files[0]["time"]]),
            dtype=torch.float32,
        ).reshape(1, signal_length)

        powers = torch.tensor(
            np.array([EpilepsyDataset._get_signal_power(x) for x in signals]),
            dtype=torch.float32,
        ).view(len(self._signals_names), signal_length)

        concatenated_signals = torch.tensor(
            np.concatenate([signals, powers, encoded_hours], axis=0)
        ).view(self._number_of_features, signal_length)

        item = torch.transpose(concatenated_signals, 0, 1)

        return item, label
