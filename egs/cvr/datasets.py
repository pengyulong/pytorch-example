import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import pandas as pd


class TableData(Dataset):

    def __init__(self,sparse_features,dense_features,label_index):
        self.sparse_features = sparse_features
        self.dense_features = dense_features
        self.label_index = label_index

    def __getitem__(self,index):
        pass

    @classmethod
    def from_csv(csvfile):
        dataSet = pd.read_csv(csvfile)
