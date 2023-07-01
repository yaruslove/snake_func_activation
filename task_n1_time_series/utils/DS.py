from torch.utils.data import Dataset

import torchvision
import torch
import numpy as np
from PIL import Image
import glob

from sklearn.preprocessing import StandardScaler


class DS(Dataset):
    def __init__(self,
                 x_data: torch.Tensor,
                 y_target: torch.Tensor,
                 time_index:np.ndarray):

        self.x_data = x_data
        self.y_target = y_target
        self.time_index=time_index

    def __getitem__(self, idx: int): #

        return self.x_data[idx], self.y_target[idx], self.time_index[idx]

    def __len__(self) -> int:
        return len(self.y_target)


def pd2tensor(data):
    data=torch.tensor(data.values.astype(float))
    data=data.to(torch.float)
    return data

def time2int(tm):
    tm= ((tm - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 'h')).astype(int)
    return tm

def int2time(tm):
    tm= tm*np.timedelta64(1, 'h') + np.datetime64('1970-01-01T00:00:00Z')
    return tm


def train_test(data, test_size = 0.15, scale = False, cols_to_transform=None, include_test_scale=False):
    """
    
        Perform train-test split with respect to time series structure
        
        - df: dataframe with variables X_n to train on and the dependent output y which is the column 'SDGE' in this notebook
        - test_size: size of test set
        - scale: if True, then the columns in the -'cols_to_transform'- list will be scaled using StandardScaler
        - include_test_scale: If True, the StandardScaler fits the data on the training as well as the test set; if False, then
          the StandardScaler fits only on the training set.
        
    """
    df = data.copy()
    # get the index after which test set starts
    test_index = int(len(df)*(1-test_size))
    
    # StandardScaler fit on the entire dataset
    if scale and include_test_scale:
        scaler = StandardScaler()
        df[cols_to_transform] = scaler.fit_transform(df[cols_to_transform])
        
    X_train = df.drop('SDGE', axis = 1).iloc[:test_index]
    y_train = df.SDGE.iloc[:test_index]
    X_test = df.drop('SDGE', axis = 1).iloc[test_index:]
    y_test = df.SDGE.iloc[test_index:]
    
    # StandardScaler fit only on the training set
    if scale and not include_test_scale:
        scaler = StandardScaler()
        X_train[cols_to_transform] = scaler.fit_transform(X_train[cols_to_transform])
        X_test[cols_to_transform] = scaler.transform(X_test[cols_to_transform])
    
    return X_train, X_test, y_train, y_test