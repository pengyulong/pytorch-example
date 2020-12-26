import torch.functional as F
from sklearn import preprocessing
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,precision_score,recall_score,classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader,RandomSampler,SequentialSampler)
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data.dataloader import default_collate
import utils
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from torch.optim import lr_scheduler
import pandas as pd
from sklearn.manifold import TSNE
from attention_dpcnn_bce import SequenceClassify


class SequenceData(Dataset):
    def __init__(self,dataX1,dataX2,dataX3,dataX4,dataX5,dataX6,dataX7,dataY):
        self.dataX1 = torch.Tensor(dataX1)
        self.dataX2 = torch.Tensor(dataX2)
        self.dataX3 = torch.Tensor(dataX3)
        self.dataX4 = torch.Tensor(dataX4)
        self.dataX5 = torch.Tensor(dataX5)
        self.dataX6 = torch.Tensor(dataX6)
        self.dataX7 = torch.Tensor(dataX7)
        self.dataY = torch.LongTensor(dataY)

    def __len__(self):
        return len(self.dataY)

    def __getitem__(self,index):
        return self.dataX1[index],self.dataX2[index],self.dataX3[index],self.dataX4[index],self.dataX5[index],self.dataX6[index],self.dataX7[index],self.dataY[index]

def load_dataSet(inputX,target):
    trainX,testX,trainY,testY = train_test_split(inputX,target,test_size=0.1,random_state=0)
    return trainX,trainY,testX,testY

def get_data(csvfile):
    dataSet = pd.read_csv(csvfile,index_col=0,header=None)
    dataX = np.array(dataSet.iloc[:,:-1])
    seq_length = dataX.shape[1]
    dataX = normalize(dataX,axis=1,norm='max')
    dataY = np.array(dataSet.iloc[:,-1]-1)
    # dataY = dataY.reshape(-1,1)
    trainX,trainY,testX,testY = load_dataSet(dataX,dataY)
    return trainX,trainY,testX,testY,seq_length

def show_feature(feature,labels,fig_name):
    x_embed = TSNE(n_components=3).fit_transform(feature)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    print(labels==1)
    x_embed_0 = x_embed[labels==0,:] # 类别0
    x_embed_1 = x_embed[labels==1,:] # 类别1
    x_embed_2 = x_embed[labels==2,:] # 类别2
    p0 = ax.scatter(x_embed_0[:,0],x_embed_0[:,1],x_embed_0[:,2],color='r',cmap='brg',marker='o',depthshade=False,label='class 0')
    p1 = ax.scatter(x_embed_1[:,0],x_embed_1[:,1],x_embed_1[:,2],color='g',cmap='brg',marker='o',depthshade=False,label='class 1')
    p2 = ax.scatter(x_embed_2[:,0],x_embed_2[:,1],x_embed_2[:,2],color='b',cmap='brg',marker='o',depthshade=False,label='class 2')
    ax.legend()
    plt.savefig(fig_name, bbox_inches='tight',dpi=300)


if __name__ == "__main__":
    trainX1,trainY,testX1,testY,seq_length1 = get_data(r"train_data/主机电流样本.csv")
    trainX2,trainY,testX2,testY,seq_length2 = get_data(r"train_data/负压样本.csv")
    trainX3,trainY,testX3,testY,seq_length3 = get_data(r"train_data/料浆样本.csv")
    trainX4,trainY,testX4,testY,seq_length4 = get_data(r"train_data/喂煤样本.csv")
    trainX5,trainY,testX5,testY,seq_length5 = get_data(r"train_data/窑头温度样本.csv")
    trainX6,trainY,testX6,testY,seq_length6 = get_data(r"train_data/窑尾温度样本.csv")
    trainX7,trainY,testX7,testY,seq_length7 = get_data(r"train_data/一次风样本.csv")
    seq_lengths = [seq_length1,seq_length2,seq_length3,seq_length4,seq_length5,seq_length6,seq_length7]
    model_file = r"result_129/BCE/best.pth.tar" #记得对应修改
    model = SequenceClassify(out_channels=129,num_class=3,hidden_num=10,seq_lengths=seq_lengths)
    valid_data = SequenceData(testX1,testX2,testX3,testX4,testX5,testX6,testX7,testY)
    valid_dataloader = DataLoader(dataset=valid_data,batch_size=len(valid_data),shuffle=False,num_workers=0,drop_last=False)

    utils.load_checkpoint(model_file,model)
    model.eval()
    for batch in valid_dataloader:
        x1,x2,x3,x4,x5,x6,x7,y = batch
        f1 = model.feature_layer1(x1).squeeze(2).detach().numpy()
        f2 = model.feature_layer2(x2).squeeze(2).detach().numpy()
        f3 = model.feature_layer3(x3).squeeze(2).detach().numpy()
        f4 = model.feature_layer4(x4).squeeze(2).detach().numpy()
        f5 = model.feature_layer5(x5).squeeze(2).detach().numpy()
        f6 = model.feature_layer6(x6).squeeze(2).detach().numpy()
        f7 = model.feature_layer7(x7).squeeze(2).detach().numpy()
        y = y.detach().numpy()
        show_feature(x1,y,"主机电流_init.png")
        show_feature(f1,y,"主机电流_cnn.png")
        show_feature(x2,y,"负压样本_init.png")
        show_feature(f2,y,"负压样本_cnn.png")
        show_feature(x3,y,"料浆样本_init.png")
        show_feature(f3,y,"料浆样本_cnn.png")
        show_feature(x4,y,"喂煤样本_init.png")
        show_feature(f4,y,"喂煤样本_cnn.png")
        show_feature(x5,y,"窑头温度样本_init.png")
        show_feature(f5,y,"窑头温度样本_cnn.png")
        show_feature(x6,y,"窑尾温度样本_init.png")
        show_feature(f6,y,"窑尾温度样本_cnn.png")
        show_feature(x7,y,"一次风样本_init.png")
        show_feature(f7,y,"一次风样本_cnn.png")




