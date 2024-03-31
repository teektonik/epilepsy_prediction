import os
import torch
import numpy as np
import pandas as pd
import csv
import glob
from tqdm import tqdm
import re 
from sklearn.utils import resample


def find_holes(df, start = 0 , diff_value=7.8125):
    # Calculate the difference between current and previous timestamp
    time_distance = df.loc[start:, 'time'].diff()
    time_distance[start] = diff_value

    # Find where the difference is not equal to diff_value
    holes =  time_distance != diff_value
    
    last_hole_index = holes[holes].last_valid_index()
    return last_hole_index

class DatasetFormatter:
    """
    Load all segments, merges it into one array, split it on EQUAL segments and saves on disk.
    Creates annotation file.
    """
    
    def __init__(self, path_to_dataset: str, 
                 verbose: bool=True, 
                 path_to_save: str="/workspace/new_data/", 
                 path_to_save_normalization: str="/workspace/normalized_data/",
                 frequency: int=128, 
                 segment_time: int=180,
                 use_data_hour_before_epilepsy: bool=False,
                 time_stamp:int=10000):
        
        if os.path.exists(path_to_dataset) is False:
            raise ValueError('There is no such path')
            
        self.segment_time = segment_time
        self.path_to_save = path_to_save
        self.path = path_to_dataset
        self.folders_with_patients = os.listdir(self.path)
        self.path_to_save_normalization = path_to_save_normalization
        self.verbose = verbose
        self.frequency = frequency
        self.patients_data = []
        self.use_data_hour_before_epilepsy = use_data_hour_before_epilepsy
        self.time_stamp = time_stamp
        
        for patient in self.folders_with_patients:
            self.patients_data.append(os.listdir(os.path.join(self.path + patient)))
            
    def get_patients_data(self, patient: str) -> list[str]:
        return os.listdir(os.path.join(self.path, patient))
    
    def get_patients_names(self) -> list[str]:
        return self.folders_with_patients
    
    def _get_only_one_sensors(self, sensors: list, sensor_name: str):
        return list(filter(lambda x: sensor_name in x, sensors))
    
    def get_all_sensors_records_for_patient(self, patient: str) -> list[str]:
        if patient not in self.get_patients_names():
            raise ValueError('There is no such name')
         
        full_path_to_patient = os.path.join(self.path, patient) 
        
        return [name for name in os.listdir(full_path_to_patient) 
                if os.path.isdir(os.path.join(full_path_to_patient, name)) 
                and name[0] != '.']
    
    def _get_sorted_segments(self, sensor_folder: str) -> (list, list[list]):
        """
        Get types of sensors from folder and sort its data by number of segment
        """
        
        delimiter = '_'
        full_path = os.path.join(self.path, sensor_folder)
        sensors_files_names = [file.split(delimiter) for file in os.listdir(full_path)]

        cropped_sensors_files_names = [file[:-2] for file in sensors_files_names]
        
        index_of_parameters = 3
        unique_parameters = set(file[index_of_parameters] for file in cropped_sensors_files_names)

        return_data = []
        for unique_item in unique_parameters:
            data = [file for file in sensors_files_names if unique_item in file]
            index_of_segments_number = 5
            sorted_list = sorted(data, key=lambda x: int(x[index_of_segments_number].split('.')[0]))
            return_data.append(['_'.join(file) for file in sorted_list])
    
        return unique_parameters, return_data
    
    def _upsample(self, data: np.array, sample_rate: float, new_sample_rate: float, mode: str = 'bicubic'):
        scale_factor = new_sample_rate / sample_rate
        upsampler = nn.Upsample(scale_factor, mode)
        return upsampler(data)
    
    
    def preprocess(self, path_to_save: str, segment_time: int=180):
        """
        Creates segments with equal size
        """
        patients = self.get_patients_names()
        self.segment_time = segment_time
        self.path_to_save = path_to_save
        
        #Create dataframe for segment labels
        columns = ['Patient', 'Segment', 'Label', 'Files']
        label_df = pd.DataFrame(columns=columns)
        
        for patient in patients:
            #Get metadata ablout epileptic events
            filename = os.path.join(self.path, patient, patient + '_' + "labels.csv")
            labels_file = pd.read_csv(filename)
            
            current_folder = os.path.join(self.path, patient)

            sensors = self.get_all_sensors_records_for_patient(patient)
            sensors = self._get_only_one_sensors(sensors, 'Empatica')
            
            if len(sensors) == 0:
                continue
            
            sensors_signals_names = []
            sensors_data_segments = []
            for sensor in sensors:
                signals_names, records = self._get_sorted_segments(os.path.join(patient, sensor))
                
                full_path_records = []
                for signal_name, sensor_record in zip(signals_names, records):
                    full_path_records.append(list(map(lambda x: os.path.join(self.path, patient, sensor, x), sensor_record)))                                
                    sensors_data_segments.append(full_path_records)
                    sensors_signals_names.append(signal_name)
                    
            data = zip(sensors_signals_names, sensors_data_segments)    
            data_size = len(sensors_signals_names)
            concatenated_df_signals = []
            segment_length = self.segment_time * self.frequency
            
            #Concated patient's signals 
            for signal_name, sensor_data in tqdm(data, total=data_size, desc='Patient: {}'.format(patient)):
                dfs = [pd.read_parquet(segment) for segment in sensor_data]
                concatenated_df = pd.concat(dfs, ignore_index=True)
                concatenated_df_signals.append(concatenated_df)
            
            starts = [0] * len(concatenated_df_signals)
            segment_index = 0
            no_hole_detected = True
            num_lenght = [len(concatenated_df) // segment_length for concatenated_df in concatenated_df_signals]
            
            pbar = tqdm(total=max(num_lenght), desc= patient + " Signals segmentation")
            while all(starts[i] + segment_length < len(concatenated_df) 
                      for i, concatenated_df in enumerate(concatenated_df_signals)):
                while not all(concatenated_df.iloc[starts[i]]['time'] == concatenated_df_signals[0].iloc[starts[0]]['time'] 
                          for i, concatenated_df in enumerate(concatenated_df_signals)):
                    
                    max_index = max(enumerate([concatenated_df.iloc[starts[i]]['time'] 
                                    for i, concatenated_df in enumerate(concatenated_df_signals)]), key=lambda x: x[1])[0]
                    
                    starts = [start + int((concatenated_df_signals[max_index].iloc[starts[max_index]]['time'] -
                                           concatenated_df_signals[i].iloc[start]['time'])//7.8125) for i, start in enumerate(starts)]
                    
                    if any(starts[j] + segment_length >= len(concatenated_df) for j, 
                           concatenated_df in enumerate(concatenated_df_signals)):
                        break
                        
                if any(starts[i] + segment_length >= len(concatenated_df) 
                       for i, concatenated_df in enumerate(concatenated_df_signals)):
                    break
                parts = []

                for i, concatenated_df in enumerate(concatenated_df_signals):
                    part = concatenated_df.iloc[starts[i] : starts[i] + segment_length]
                    hole_index = find_holes(part, starts[i])
                    if hole_index is not None:
                        starts[i]= hole_index
                        no_hole_detected = False
                        continue 
                    parts.append(part)
                
                segment_end =  parts[0]['time'].iloc[len(parts[0])-1]
                segment_start =  parts[0]['time'].iloc[0]
                
                label = 0
                data_before_epilepsy = False
                for j in range(len(labels_file)):
                    if (segment_start < labels_file['labels.startTime'][j] + labels_file['labels.duration'][j] <= segment_end 
                        or segment_start < labels_file['labels.startTime'][j] <= segment_end):
                        label = 1
                    if (segment_start < labels_file['labels.startTime'][j] - self.time_stamp <= segment_end and self.use_data_hour_before_epilepsy):
                        data_before_epilepsy = True
                        
                if label == 1:
                    data_before_epilepsy = False
                
                if no_hole_detected and not (self.use_data_hour_before_epilepsy and not data_before_epilepsy):
                    for i, part in enumerate(parts):
                        file_path = os.path.join(self.path_to_save, '_'.join([patient, sensors_signals_names[i], str(segment_index)]) + '.parquet')
                        mask = (label_df['Patient'] == patient) & (label_df['Segment'] == segment_index)
                    
                        if label_df[mask].any().any():
                            #if the row exists, append the new file paths to the existing ones
                            existing_files = label_df.loc[mask, 'Files'].values[0]

                            if pd.isna(existing_files):
                                #if the existing value is NaN, replace it with the new file paths
                                label_df.loc[mask, 'Files'] = file_path
                            else:
                                # otherwise, append the new file paths
                                label_df.loc[mask, 'Files'] = existing_files + ', ' + file_path
                        else:
                            #Label calculation/set
                            segment_end =  part['time'].iloc[len(part)-1]
                            segment_start =  part['time'].iloc[0]

                            new_row = pd.DataFrame([[patient, segment_index, label, file_path]], columns=columns)
                            label_df = pd.concat([label_df, pd.DataFrame(new_row)], ignore_index=True)

                        part.to_parquet(file_path, index=False)
                    segment_index += 1 
                    starts = [start + segment_length for start in starts]
                no_hole_detected = True
                pbar.update(1)
            pbar.close()
        label_df.to_csv(os.path.join('/workspace', 'labels.csv'), index=False)