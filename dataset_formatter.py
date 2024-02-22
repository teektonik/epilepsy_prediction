import os
import torch
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm


class DatasetFormatter:
    """
    Load all segments, merges it into one array, split it on EQUAL segments and saves on disk.
    Creates annotation file.
    """
    
    def __init__(self, path_to_dataset: str, verbose: bool=True):
        if os.path.exists(path_to_dataset) is False:
            raise ValueError('There is no such path')
            
        self.path = path_to_dataset
        self.folders_with_patients = os.listdir(self.path)
        self.verbose = verbose
        self.segment_time = 0
        self.frq = 60
        self.path_to_save = ""
        self.patients_data = []
        for patient in self.folders_with_patients:
            self.patients_data.append(os.listdir(os.path.join(self.path + patient)))
            
    def get_patients_data(self, patient: str) -> list[str]:
        return os.listdir(os.path.join(self.path, patient))
    
    def get_patients_names(self) -> list[str]:
        return os.listdir(self.path)
    
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
        for patient in patients:
            current_folder = os.path.join(self.path, patient)
            
            label_file = os.path.join(current_folder,
                                      [filename for filename in os.listdir(current_folder) if 'labels' in filename][0])
            
            labels = pd.read_csv(label_file)
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
           
            for signal_name, sensor_data in tqdm(data, total=data_size, desc='Patient: {}'.format(patient)):
                dfs = [pd.read_parquet(segment) for segment in sensor_data]
                concatenated_df = pd.concat(dfs, ignore_index=True)
                
                num_parts = int(len(concatenated_df) / (self.segment_time * self.frq))
         
                parts = np.array_split(concatenated_df, num_parts)
            
                for i, part in enumerate(parts):
                    part.to_parquet(os.path.join(self.path_to_save, '_'.join([patient, signal_name, str(i)]) + '.parquet'),
                                                 index=False)
            
                
    def labels_set(self):
        """
        Get .cvs metadata and calculate labels for each segment from each Patient
        """
        if(self.segment_time == 0):
            raise ValueError("self.segment_time is 0. This mean there have not been segmentation process for signals yet.")
        segment_length = self.segment_time * self.frq
        
        for patient in self.folders_with_patients:
            filename = patient+"labels.csv"
            labels_file = pd.read_csv(filename)
            duration = labels_file['duration'].iloc[0]
            start_time = labels_file['startTime'].iloc[0]
            total_segments = int(duration.total_seconds() / (segment_length))
            labels = []
            for i in range(total_segments):
                segment_start = start_time + i * segment_length
                segment_end = segment_start + segment_length
                labels.append(0)
                for j in range(len(labels_file)):
                    if (labels_file['labels.startTime'][j] + labels_file['labels.duration'][j] > segment_start and 
                        labels_file['labels.startTime'][j] + labels_file['labels.duration'][j] <= segment_end):
                        labels[-1] = 1
            full_path = os.path.join(self.path_to_save, filename)
            with open(full_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['segment', 'label'])  # writing the headers
                for idx, label in enumerate(labels):
                    writer.writerow([idx, label])
    def simple_normilization(self, segment_file: str):
        """
        Take .parquet files normilize them with simple algorithm
        """
        for patient in self.folders_with_patients:
            full_path = os.path.join(self.path_to_save, '_'.join([patient, signal_name, str(i)]) + '.parquet'
            part = pd.read_parquet(full_path)

            mean = part['data'].mean()
            std_dev = part['data'].std()
            part['data'] = (part['data'] - mean) / std_dev

            part.to_parquet(full_path, index=False)
