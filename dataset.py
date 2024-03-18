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
                 signal_lenght: int):
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
    
        # Class balancing
        class_0 = self._annotation_file[self._annotation_file['Label'] == 0]
        class_1 = self._annotation_file[self._annotation_file['Label'] == 1]

        num_samples_to_keep = min(len(class_0), len(class_1))
        self._annotation_file = pd.concat([class_0.sample(num_samples_to_keep, random_state=42), class_1], axis=0)

    @staticmethod    
    def _get_acc_sqi(acc_x, acc_y, acc_z, sampling_rate=128):
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

            f_acc, Pxx_acc = periodogram(segment_data, fs=sampling_rate)

            idx_08hz = np.argmax(f_acc >= 0.8)
            idx_5hz = np.argmax(f_acc >= 5.0)

            narrowband_power = np.mean(Pxx_acc[idx_08hz:idx_5hz + 1])
            broadband_power = np.mean(Pxx_acc[idx_08hz:])

            narrowband_powers.append(narrowband_power)
            broadband_powers.append(broadband_power)
    
        avg_narrowband_power = np.mean(narrowband_powers)
        avg_broadband_power = np.mean(broadband_powers)

        signal_quality = avg_narrowband_power / avg_broadband_power
 
        return signal_quality
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Number of samples.
        """
        return len(self._annotation_file)
    
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
        signals = np.array([x['data'] / float(10 ** 9) for x in files])
        
        ms_in_hour = 3_600_000
        encoded_hours = ((files[0]['time'][0] / ms_in_hour) % 24) / 24
        encoded_hours = torch.tensor([encoded_hours] * self.signal_length, dtype=torch.float32)
        
        concated_signals = torch.tensor(np.concatenate([signals, encoded_hours.reshape(1, len(encoded_hours))], axis=0)) \
                                                                .view(len(self._signals_names) + 1, self.signal_length)
        
        item = torch.cat([concated_signals.unsqueeze(0)], dim=0).view(len(self._signals_names) + 1, self.signal_length)
        item = torch.transpose(item, 0, 1)
        
        # Return the concatenated array as the sample item and its label
        return item, label
