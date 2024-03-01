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
        
        signals = np.array([pd.read_parquet(path)['data'][:self.signal_length] for path in paths_to_segments])
        spectrums = np.array([np.array(np.fft.fft(signal).real[:self.signal_length], dtype=np.float32) for signal in signals], dtype=np.float32)
        
        concated_signals = torch.tensor(np.concatenate([signals], axis=0)).view(8, self.signal_length)
        concated_spectrums = torch.tensor(np.concatenate([spectrums], axis=0)).view(8, self.signal_length)
        
        item = torch.cat([concated_signals.unsqueeze(0), concated_spectrums.unsqueeze(0)], dim=0).view(16, self.signal_length)
        item = torch.transpose(item, 0, 1)
        
        # Return the concatenated array as the sample item and its label
        return item, label
