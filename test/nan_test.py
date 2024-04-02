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
    files = [pd.read_parquet(path) for path in paths_to_segments]
    for file in files:
        feature = (file['data'].to_numpy()) 
        if (np.isnan(feature).any()):
            return 1 #if NaN
    return 0
    
def test_files():
    """
    Test function to validate the files data processing
    """
    
    dataframe = pd.read_csv('/workspace/labels.csv') # Loading the labels data
    segments = dataframe["Segment"].to_numpy()
    
    for el in segments:
        check = files(dataframe, el)
        # Asserting that the data don't сontsins NaN
        assert check == 0
