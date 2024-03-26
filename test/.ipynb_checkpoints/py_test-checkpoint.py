import pytest
import pandas as pd
import os
import numpy as np

def files(dataframe, idx: int):
     """
     Function to process files based on the given index
     Extracting necessary data from the dataframe
     """
    data_row = dataframe.iloc[idx]
    path_to_data = '/workspace/new_data/'
    patient_name = data_row['Patient']
    segment_id = str(data_row['Segment'])
    signals_names = ['Acc x', 'Acc y', 'Acc z', 'Acc Mag', 'EDA', 'BVP', 'TEMP', 'HR']
    # Constructing paths to segments based on patient name, signal, and segment ID
    paths_to_segments = [
        "_".join(
            [os.path.join(path_to_data, patient_name), signal, segment_id]
        ) + ".parquet"
        for signal in signals_names
    ]
    # Reading data files based on the constructed paths
    files = [pd.read_parquet(path) for path in paths_to_segments]
    # Extracting necessary information from each file
    start_time = [file.loc[0, 'time'] for file in files]
    len_of_files = [len(file['time']) for file in files]
    end_of_files = [file.loc[len(file['time'])-1, 'time'] for file in files]
        
    return start_time, len_of_files, end_of_files
    
def test_files():
    """
    Test function to validate the files data processing
    """
    
    dataframe = pd.read_csv('/workspace/labels.csv') # Loading the labels data
    segments = dataframe["Segment"].to_numpy()
    
    for el in segments:
        start_time, len_of_files, end_of_files = files(dataframe, el)
        # Asserting that the start time, length, and end time are consistent across files
        assert len(np.unique(start_time)) == 1
        assert len(np.unique(len_of_files)) == 1
        assert len(np.unique(end_of_files)) == 1
