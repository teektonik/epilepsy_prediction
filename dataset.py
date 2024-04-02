# Epileptic Dataset

import os
import numpy as np
import pandas as pd
import scipy
from scipy.signal import periodogram
from scipy.stats import iqr
from mne.time_frequency import psd_array_multitaper
from scipy.signal import butter, filtfilt

import torch
from torch.utils.data import Dataset

from model_arguments import ModelArguments

class BaseDataset(Dataset):
    """
    A base class containing common code for others
    """
    def __init__(
        self, 
        args: ModelArguments
    ):
        """
        Initializes the BaseDataset with the given parameters.

        Parameters:
        - args (ModelArguments): A class storing all parameters.
        """
        super().__init__()
        self._path_to_data = args.path_to_data
        self._signals_names = args.signals_names
        self._number_of_features = args.number_of_channels
        
        self._annotation_file = pd.read_csv(args.path_to_annotation)

        # Class balancing
        if args.use_class_balance:
            class_0 = self._annotation_file[self._annotation_file["Label"] == 0]
            class_1 = self._annotation_file[self._annotation_file["Label"] == 1]

            num_samples_to_keep = min(len(class_0), len(class_1))
            self._annotation_file = pd.concat(
                [class_0.sample(num_samples_to_keep, random_state=42), class_1], axis=0)

    @staticmethod
    def _apply_lowpass_filter(data: np.array, cutoff_frequency: int, fs: int, order: int=5) -> np.array:
        """
        Apply a lowpass filter to the data.

        Parameters:
        - data: The input signal.
        - cutoff_frequency: The cutoff frequency of the filter.
        - fs: The sampling frequency of the data.
        - order (int): The order of the filter (default: 5).

        Returns:
        - filtered_data: The filtered signal.
        """
        nyq = 0.5 * fs
        normal_cutoff = cutoff_frequency / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = filtfilt(b, a, data)
        
        return filtered_data
        
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Number of samples.
        """
        return len(self._annotation_file)

    @staticmethod
    def _signal_normalization(signal):
        """
        Simpe normalization of input signal.

        Parameters:
        - signal (np.array): Input signal.
        
        Returns:
        - norm_signal (np.array): normalized signal.
        """
        mean = signal.mean()
        std_dev = signal.std()
        if std_dev == 0:
            return signal
        norm_signal = (signal - mean) / std_dev
        
        return norm_signal
        
    @staticmethod
    def _signal_noise_augmentation(signal):
        """
        Simpe normalization of input signal. 

        Parameters:
        - signal (np.array): Input signal.
        
        Returns:
        - noise_signal(np.array): signal with added noise. 
        """
        median_val = np.median(signal)

        noise = np.random.normal(0, abs(median_val), len(signal))
        noise_signal = signal + noise
        return noise_signal

    def __getitem__(self, idx: int) -> tuple:
        """
        Gets the data for a given index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - tuple: Concatenated array of signals and their spectra, and the item's label.
        """
        data_row = self._annotation_file.iloc[idx]
        patient_name = data_row['Patient']
        segment_id = str(data_row['Segment'])
        label = data_row['Label']

        paths_to_segments = [
            "_".join(
                [os.path.join(self._path_to_data, patient_name), signal, segment_id]
            )
            + ".parquet"
            for signal in self._signals_names
        ]

        files = [pd.read_parquet(path) for path in paths_to_segments]
        
        signals = [x['data'] / float(10 ** 9) for x in files]
        time = [x['time'] for x in files]
        

        if self.args.use_normalization:
            signals = [EpilepsyDataset._signal_normalization(signal) for signal in signals]
            
        if self.args.use_noise_augmentation:
            signals = [EpilepsyDataset._signal_noise_augmentation(signal) if np.random.rand() < self.args.noise_chance_augmentation else signal
                       for signal in signals]
            
        signals = [signal.astype(np.float32) for signal in signals]
        return signals, time, label

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

class RawEpilepsyDataset(BaseDataset):
    """
    Dataset class for handling raw epileptic data.
    """

    def __init__(
        self, 
        args: ModelArguments
    ):
        """
        Initializes the EpilepsyDataset with the given parameters.

        Parameters:
        - args (ModelArguments): A class storing all parameters
        """
        super().__init__(args)
        self.args = args

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
        Calculate the power of a signal.

        Parameters:
        - signal (np.array): Input signal.
        - sample_rate (int): Sampling rate of the signal.

        Returns:
        - np.array: Power spectrum of the signal.
        """
        power = np.power(np.abs(np.fft.fft(signal)), 2)
        return power.astype(float)
    
    @staticmethod
    def _get_spectrum(signal):
        """
        Calculate the spectrum of a signal and save it as matrix of magnitude and phase.

        Parameters:
        - signal (np.array): Input signal.
        
        Returns:
        - np.array: matrix of concated magnitude and phase.
        """
        fft_vals = np.fft.fft(signal)
        magnitude = np.abs(fft_vals)
        phase = np.angle(fft_vals)
        matrix = np.vstack([magnitude, phase])

        return matrix
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Gets the data for a given index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - tuple: Concatenated array of signals and their spectra, and item's label.
        """
        signals, time, label = super().__getitem__(idx)
        signal_length = len(signals[0])

        ms_in_hour = 3_600_000
        encoded_hours = torch.tensor(
            np.array([(((x / ms_in_hour) % 24) / 24) for x in time[0]]),
            dtype=torch.float32,
        ).reshape(1, signal_length)
        
        '''
        powers = torch.tensor(
            np.array([RawEpilepsyDataset._get_signal_power(x) for x in signals]),
            dtype=torch.float32,
        ).view(len(self._signals_names), signal_length)
        '''

        spectrums = torch.tensor(
            np.array([self._get_spectrum(signal) for signal in signals]), 
            dtype=torch.float32
        ).view(-1, signal_length)
    
        if self.args._use_spectrum:
            concatenated_signals = torch.tensor(
                np.concatenate([signals, encoded_hours, spectrums], axis=0)
            ).view(self._number_of_features, signal_length)
        else:
            concatenated_signals = torch.tensor(
                np.concatenate([signals, encoded_hours], axis=0)
            ).view(self._number_of_features, signal_length)


        item = torch.transpose(concatenated_signals, 0, 1)

        return item, label
    

class EpilepsyDataset(BaseDataset):
    """
    Dataset class for handling epileptic data with extracted features.
    """

    def __init__(
        self, 
        args: ModelArguments
    ):
        """
        Initializes the EpilepsyDataset with the given parameters.

        Parameters:
        - args (ModelArguments): A class storing all parameters.
        """
        super().__init__(args)
        self.args = args
        
    @staticmethod
    def _get_statistics(time_domain_data: np.array, window_size_in_elements: int) -> tuple:
        """
        Calculate statistical features from time domain data.

        Parameters:
        - time_domain_data (np.array): Data in the time domain.
        - window_size_in_elements (int): Size of the window in elements.

        Returns:
        - tuple: Mean, standard deviation and interquartile range of the data.
        """
        number_of_windows = int(len(time_domain_data) / window_size_in_elements)
        windows = np.array_split(time_domain_data, number_of_windows)
        
        mean = np.mean(windows, axis=1).reshape(1, number_of_windows)
        std = np.std(windows, axis=1).reshape(1, number_of_windows)
        iqr =  scipy.stats.iqr(windows, axis=1).reshape(1, number_of_windows)

        return mean, std, iqr
        
    @staticmethod
    def _get_spectral_features(time_domain_data: np.array, sample_rate: int = 128, window_size: int=4
) -> tuple:
        """
        Calculate spectral features from frequency domain data.

        Parameters:
        - time_domain_data (np.array): Data in the time domain.
        - sample_rate (int): Sampling rate of the data.
        - window_size (int): Size of the window.

        Returns:
        - tuple: Mean, standard deviation and iqr of the data.
        """
        spectrum = np.fft.fft(time_domain_data)
        spectrum = BaseDataset._apply_lowpass_filter(spectrum, 50, 128)
        power = np.power(np.abs(spectrum), 2)
    
        return EpilepsyDataset._get_statistics(power, sample_rate * window_size)

    @staticmethod
    def _get_time_features(time_domain_data: np.array, sample_rate: int = 128, window_size: int=4) -> tuple:
        """
        Calculate time-domain features.

        Parameters:
        - time_domain_data (np.array): Data in the time domain.
        - sample_rate (int): Sampling rate of the data.
        - window_size (int): Size of the window.

        Returns:
        - tuple: Mean, standard deviation and iqr of the data.
        """
        
        return EpilepsyDataset._get_statistics(time_domain_data, sample_rate * window_size)
    
    @staticmethod
    def _get_encoded_time(
        time_array: np.array, sample_rate: int = 128, window_in_sec: int = 4
    ) -> tuple:
        """
        Encode time data.

        Parameters:
        - time_array (np.array): Array containing time data.
        - sample_rate (int): Sampling rate of the data.
        - window_in_sec (int): Size of the window in seconds.

        Returns:
        - tuple: Encoded time data.
        """
      
        number_of_windows = int(len(time_array) / (window_in_sec * sample_rate))

        ms_in_hour = 3_600_000
        encoded_hours = np.array(
            [
                (((time_array[i * (window_in_sec * sample_rate)] / ms_in_hour) % 24) / 24)
                for i in range(number_of_windows)
            ]
        ).reshape(1, number_of_windows)

        return encoded_hours

    @staticmethod
    def _compute_zero_crossing_rate(signals):
        """
        Compute the zero-crossing rate of the signals.

        Parameters:
        - signals (np.array): The input signals.

        Returns:
        - np.array: The zero-crossing rate of the signals.
        """
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signals), axis=1)) / 2, axis=1)
        return zero_crossings

    def __getitem__(self, idx: int) -> tuple:
        """
        Gets the data for a given index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - tuple: Concatenated array of signals and their spectra, and the item's label.
        """
        signals, time, label = super().__getitem__(idx)
        signal_length = len(signals[0])

        time_features_list = [
            np.concatenate(
                EpilepsyDataset._get_time_features(
                    x, self.args.sample_rate, self.args.window_size
                ), axis=0
            ) for x in signals
        ]
        
        frequency_features_list = [
            np.concatenate(
                EpilepsyDataset._get_spectral_features(
                    x, self.args.sample_rate, self.args.window_size
                ), axis=0
            ) for x in signals
        ]
        
        time_features = torch.tensor(np.concatenate(time_features_list, axis=0), dtype=torch.float32)
        frequency_features = torch.tensor(np.concatenate(frequency_features_list, axis=0), dtype=torch.float32)
        encoded_time = torch.tensor(EpilepsyDataset._get_encoded_time(time[0]), dtype=torch.float32)

        concatenated_signals = torch.tensor(
            np.concatenate([time_features, frequency_features, encoded_time], axis=0)
        ).view(self._number_of_features, int(signal_length / self.args.window_size_in_elements))

        item = torch.transpose(concatenated_signals, 0, 1)

        return item, label
