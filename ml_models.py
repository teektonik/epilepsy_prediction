# Models

import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from dataset import EpilepsyDataset 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier

class BaselineModel(nn.Module):
    def __init__(self, input_size: int, hidden_nodes: int=128, lstm_layers: int=4):
        super(BaselineModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_nodes, lstm_layers, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_nodes, 2)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.rnn(x)
        lstm_out = lstm_out[:, -1, :]
        output = self.sigmoid(self.fc(lstm_out))
        
        return output


class MachineLearning():
    def __init__(self, dataset_path_label: str, normal_data_path: str, list_of_signals: list[str], signal_lenght: int):
        '''
        Initialize the feature dataframe 
        
        Parameters:
        - dataset_path_label (str): Path to the annotation file.
        - normal_data_path (str): Path to the data directory.
        - list_of_signals (list[str]): List of signal names.
        - signal_length (int): lenght of a signal
        '''
        self.dataset_path_label = dataset_path_label
        self.normal_data_path = normal_data_path
        self.signal_length = signal_lenght
        self.data = pd.read_csv(dataset_path_label)
        self.list_of_signals = list_of_signals
        self.X, self.y = self.get_train_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.30, random_state = 2020, stratify=self.y)
           
    
    def print_report(self, y_valid, y_pred):
        '''
        A function for displaying metrics
        
        Parameters:
        - y_valid - true data
        - y_pred - model predictions
        '''
        thresh = 0.5
       
        auc = roc_auc_score(y_valid, y_pred) 
        accuracy = accuracy_score(y_valid, (y_pred > thresh))
        recall= recall_score(y_valid, (y_pred > thresh)) 
        precision = precision_score(y_valid, (y_pred > thresh))
        
        print('AUC: %.3f'%auc)
        print('accuracy:%.3f'%accuracy)
        print('recall:%.3f'%recall)
        print('precision:%.3f'%precision)
        print(' ')
        
    
    def get_train_data(self):
        '''
        A function to translate the dataset into dataframe so that
        machine learning methods can be trained
        '''
        idx = self.data['Segment'].to_numpy()
        x = []
        y = []
         
        train_data = EpilepsyDataset(self.dataset_path_label, 
                                     self.normal_data_path,
                                     self.list_of_signals,
                                     self.signal_length)
        for el in range (0, len(idx)):
            feature = train_data[el]
            y1 = (feature[1])
            feature = (feature[0].numpy())
            feature = feature.reshape((-1,1))
            
            #if (np.isnan(feature).any()):
            #   print('!', end=" ")
            #   continue 
            
            if el == 0:
                x = feature
                y = y1
            else:
                x = np.concatenate((x, feature), axis=1)
                y = np.concatenate((y, y1), axis=None)
            if el%100==0:
                print(el, end=' ')
        print(' ')    

        X = pd.DataFrame(x.T)
        y = y.reshape(len(y), 1)
        y = pd.DataFrame(y)
        
        return X, y[0]
    
    
    def knn(self):
        '''
        train Knn
        '''
        model = KNeighborsClassifier(n_neighbors=40, weights='distance',metric='manhattan', n_jobs=-1)
        
        model.fit(self.X_train, self.y_train)
        y_train_preds = model.predict_proba(self.X_train)[:,1]
        y_valid_preds = model.predict_proba(self.X_test)[:,1]
        print('Knn')
        print('Training:')
        self.print_report(self.y_train, y_train_preds)
        print('Test:')
        self.print_report(self.y_test, y_valid_preds)

        
    def xgb(self):
        '''
        train XGB
        '''
        xgbc = XGBClassifier(booster="gbtree", max_depth=3,
                             objective = 'binary:logistic', device = 'cuda',
                             eval_metric = "auc", colsample_bytree = 0.78)
        
        xgbc.fit(self.X_train, self.y_train)
        y_train_preds = xgbc.predict_proba(self.X_train)[:,1]
        y_valid_preds = xgbc.predict_proba(self.X_test)[:,1]
        print('xgb')
        print('Training:')
        self.print_report(self.y_train, y_train_preds)
        print('Test:')
        self.print_report(self.y_test, y_valid_preds)
    
    
    def gbc(self):
        '''
        train gbc
        '''
        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                         max_depth=3, random_state=69) 
        gbc.fit(self.X_train, self.y_train)
        y_train_preds = gbc.predict_proba(self.X_train)[:,1]
        y_valid_preds = gbc.predict_proba(self.X_test)[:,1]
        print('gbc')
        print('Training:')
        self.print_report(self.y_train, y_train_preds)
        print('Test:')
        self.print_report(self.y_test, y_valid_preds)
        
        
    def etc(self):
        '''
        train etc
        '''
        etc = ExtraTreesClassifier(bootstrap=False, criterion="entropy", 
                                   max_features=1.0, min_samples_leaf=3,
                                   min_samples_split=20, n_estimators=100)
        etc.fit(self.X_train, self.y_train)
        y_train_preds = etc.predict_proba(self.X_train)[:,1]
        y_valid_preds = etc.predict_proba(self.X_test)[:,1]
        print('etc')
        print('Training:')
        self.print_report(self.y_train, y_train_preds)
        print('Test:')
        self.print_report(self.y_test, y_valid_preds)
        