# Epileptic Dataset

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset



class EpilepsyDataset(Dataset):
    """
    Dataset class for handling epileptic data.
    """
    def __init__(self, 
                 path_to_annotation_file: str, 
                 path_to_data: str, 
                 signals_names: list[str], 
                 signal_lenght: int,
                 normalization_trigger: bool,
                 noise_augmentation_trigger: bool):
        """
        Initializes the EpilepsyDataset with the given parameters.

        Parameters:
        - path_to_annotation_file (str): Path to the annotation file.
        - path_to_data (str): Path to the data directory.
        - signals_names (list[str]): List of signal names.
        - signal_length (int): lenght of a signal
        """
        super().__init__()
        self._path_to_annotation_file = path_to_annotation_file
        self._path_to_data = path_to_data
        self._annotation_file = pd.read_csv(self._path_to_annotation_file)
        self._signals_names = signals_names
        self.signal_length = signal_lenght
        self.normalization_trigger = normalization_trigger
        self.noise_augmentation_trigger = noise_augmentation_trigger
        # Class balancing
        class_0 = self._annotation_file[self._annotation_file['Label'] == 0]
        class_1 = self._annotation_file[self._annotation_file['Label'] == 1]

        num_samples_to_keep = min(len(class_0), len(class_1))
        self._annotation_file = pd.concat([class_0.sample(num_samples_to_keep, random_state=42), class_1], axis=0)
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Number of samples.
        """
        return len(self._annotation_file)
    
    @staticmethod
    def get_spectrum(signal):
        """
        Calculate the spectrum of a signal.

        Parameters:
        - signal (np.array): Input signal.
        
        Returns:
        - np.array: matrix of concated magnitude and phase 
        """
        # Compute FFT
        fft_vals = np.fft.fft(signal)

        # Compute magnitude and phase
        magnitude = np.abs(fft_vals)
        phase = np.angle(fft_vals)

        # Create a matrix with signal, magnitude and phase as rows
        matrix = np.vstack([magnitude, phase])

        return matrix
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
    
    def __getitem__(self, idx):
        """
        Gets the data for a given index.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - torch.tensor: Concatenated array of signals and their spectrums.
        - int: Item's label
        """
        data_row = self._annotation_file.iloc[idx]
        patient_name = data_row['Patient']
        segment_id = str(data_row['Segment'])
        label = data_row['Label']
        
        paths_to_segments = ['_'.join([os.path.join(self._path_to_data, patient_name), 
                                      signal, segment_id]) + '.parquet' for signal in self._signals_names]
        files = [pd.read_parquet(path) for path in paths_to_segments]
        
        signals = np.array([pd.read_parquet(path)['data'] / float(10 ** 9) for path in paths_to_segments])
        
        if self.normalization_trigger:
            for signal in signals:
                mean = signal.mean()
                std_dev = signal.std()
                if std_dev == 0:
                    continue
                signal = (signal - mean) / std_dev
                
        if self.noise_augmentation_trigger:
            chance = 0.5
            for signal in signals:
                rand = np.random.rand()
                if rand < chance:
                    median_val = np.median(signal)

                    noise = np.random.normal(0, abs(median_val), len(signal))
                    signal = signal + noise
        
        signal_length = len(signals[0])

        ms_in_hour = 3_600_000
        encoded_hours = torch.tensor(
            np.array([(((x / ms_in_hour) % 24) / 24) for x in files[0]["time"]]),
            dtype=torch.float32,
        ).view(-1, self.signal_length)

        powers = torch.tensor(
            np.array([EpilepsyDataset._get_signal_power(x) for x in signals]),
            dtype=torch.float32,
        ).view(-1, self.signal_length)
        
        # Compute the spectrum for each signal and concatenate them
        spectrums = np.array([self.get_spectrum(signal) for signal in signals])
        spectrums = torch.tensor(spectrums).view(-1, self.signal_length)
        # Concatenate the signals and their spectrums
        concated_signals_spectrums = np.concatenate([signals, spectrums, encoded_hours], axis=0)

        # Convert to torch tensor and reshape
        item = torch.tensor(concated_signals_spectrums).view(-1, self.signal_length)
        item = torch.transpose(item, 0, 1)
        item = item.float()
        #print(item)
        # Return the concatenated array as the sample item and its label
        return item, label
