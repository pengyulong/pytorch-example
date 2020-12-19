import torch.nn.functional as F
from sklearn import preprocessing
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,classification_report
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
from tensorboardX import SummaryWriter
from model.model import SequenceClassify

utils.setup_seed(2020)

class SequenceData(Dataset):
    def __init__(self,dataX1,dataX2,dataX3,dataX4,dataX5,dataX6,dataX7,dataY):
        self.dataX1 = torch.Tensor(dataX1)
        self.dataX2 = torch.Tensor(dataX2)
        self.dataX3 = torch.Tensor(dataX3)
        self.dataX4 = torch.Tensor(dataX4)
        self.dataX5 = torch.Tensor(dataX5)
        self.dataX6 = torch.Tensor(dataX6)
        self.dataX7 = torch.Tensor(dataX7)
        one_hot = OneHotEncoder(sparse=False)
        dataY = one_hot.fit_transform(dataY)
        self.dataY = torch.Tensor(dataY)

    def __len__(self):
        return len(self.dataY)

    def __getitem__(self,index):
        return self.dataX1[index],self.dataX2[index],self.dataX3[index],self.dataX4[index],self.dataX5[index],self.dataX6[index],self.dataX7[index],self.dataY[index]


def get_data(csvfile):
    dataSet = pd.read_csv(csvfile,index_col=0,header=None)
    dataX = np.array(dataSet.iloc[:,:-1])
    seq_length = dataX.shape[1]
    dataX = normalize(dataX,axis=1,norm='max')
    dataY = np.array(dataSet.iloc[:,-1]-1)
    dataY = dataY.reshape(-1,1)
    trainX,trainY,testX,testY = load_dataSet(dataX,dataY)
    return trainX,trainY,testX,testY,seq_length

class Job(object):
    def __init__(self):
        self.device = get_device()
        self.batch_size = 128
        self.epoches = 200
        self.lr = 0.001
        self.trainX1,self.trainY,self.testX1,self.testY,seq_length1 = get_data(r"train_data/主机电流样本.csv")
        self.trainX2,self.trainY,self.testX2,self.testY,seq_length2 = get_data(r"train_data/负压样本.csv")
        self.trainX3,self.trainY,self.testX3,self.testY,seq_length3 = get_data(r"train_data/料浆样本.csv")
        self.trainX4,self.trainY,self.testX4,self.testY,seq_length4 = get_data(r"train_data/喂煤样本.csv")
        self.trainX5,self.trainY,self.testX5,self.testY,seq_length5 = get_data(r"train_data/窑头温度样本.csv")
        self.trainX6,self.trainY,self.testX6,self.testY,seq_length6 = get_data(r"train_data/窑尾温度样本.csv")
        self.trainX7,self.trainY,self.testX7,self.testY,seq_length7 = get_data(r"train_data/一次风样本.csv")
        self.seq_lengths = [seq_length1,seq_length2,seq_length3,seq_length4,seq_length5,seq_length6,seq_length7]

        self.num_class = 3
        self.out_channels = 75 #[75,150,169,207,209,129]
        self.hidden_num = 10
        self.loss_type = 'BCE' # BCE, CROSS, SMOOTH
        self.model_dir = "./result_{}/{}".format(self.out_channels,self.loss_type)
        if os.path.exists(self.model_dir)==False:
            os.makedirs(self.model_dir)
        self.log_file = utils.set_logger(r"./result_{}/{}/train.log".format(self.out_channels,self.loss_type))
        self.writer = SummaryWriter(os.path.join(self.model_dir,"runs"))
        # self.trainX,self.trainY,self.testX,self.testY = load_dataSet(self.featureX,self.targetY)

    def train(self):
        train_data = SequenceData(self.trainX1,self.trainX2,self.trainX3,self.trainX4,self.trainX5,self.trainX6,self.trainX7,self.trainY)
        valid_data = SequenceData(self.testX1,self.testX2,self.testX3,self.testX4,self.testX5,self.testX6,self.testX7,self.testY)

        train_dataloader = DataLoader(dataset=train_data,sampler=RandomSampler(train_data),batch_size=self.batch_size,shuffle=False,num_workers=0,collate_fn=default_collate,drop_last=True)

        valid_dataloader = DataLoader(dataset=valid_data,batch_size=self.batch_size,shuffle=False,num_workers=0,drop_last=False)
        model = SequenceClassify(out_channels=self.out_channels,num_class=self.num_class,hidden_num=self.hidden_num,seq_lengths = self.seq_lengths)
        model.to(device=self.device)
        optim = torch.optim.Adam(model.parameters(), lr = self.lr,betas=(0.9,0.99))

        pos_weight = torch.Tensor([0.127,0.125,0.748]).to(device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        valid_loss, train_loss = train_and_evaluate(model,train_dataloader,valid_dataloader,optim,criterion,metrics,self.epoches,model_dir=self.model_dir,restore_file=None)
        curr_hyp = {"epochs":self.epoches,"batch_size":self.batch_size,"lr":self.lr,"hidden_num":self.hidden_num,"out_channels":self.out_channels}
        utils.save_dict_to_json(curr_hyp,os.path.join(self.model_dir,"train_hyp.json"))
        valid_df = pd.DataFrame(valid_loss)
        train_df = pd.DataFrame(train_loss)
        valid_df.to_excel(os.path.join(self.model_dir,"valid_loss.xlsx"))
        train_df.to_excel(os.path.join(self.model_dir,"train_loss.xlsx"))


    def predict(self):
        valid_data = SequenceData(self.testX1,self.testX2,self.testX3,self.testX4,self.testX5,self.testX6,self.testX7,self.testY)
        valid_dataloader = DataLoader(dataset=valid_data,batch_size=len(valid_data),shuffle=False,num_workers=0,drop_last=False)

        model = SequenceClassify(out_channels=self.out_channels,num_class=self.num_class,hidden_num=self.hidden_num,seq_lengths = self.seq_lengths)
        utils.load_checkpoint(os.path.join(self.model_dir,"best.pth.tar"),model)
        # model.to(device=self.device)
        
        model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                inputX1,inputX2,inputX3,inputX4,inputX5,inputX6,inputX7,label_batch = batch
                label_batch = label_batch
                # inputX,inputY = inputX.to(device),inputY.to(device)
                y_pred = np.argmax(model(inputX1,inputX2,inputX3,inputX4,inputX5,inputX6,inputX7).data.cpu().numpy(),axis=1).squeeze()
                y_true = np.argmax(label_batch.data.cpu().numpy(),axis=1).squeeze()
                # y_true = inputY.data.cpu().numpy().squeeze()
                result = pd.DataFrame(data={'y_true':y_true,'y_pred':y_pred},index=range(len(y_pred)))
                result.to_csv(r"{}/result.csv".format(self.model_dir)) 
                # print("r2:{},rmse:{}"(y_pred,y_true),utils.rmse(y_pred,y_true)))
                evaluate_model(y_pred,y_true)

    def plot_loss(self):
        csvfile = os.path.join(self.model_dir,"loss.csv")
        assert os.path.exists(csvfile)
        fig = plt.figure(figsize=(10,8))
        data = pd.read_csv(csvfile)
        train_loss, valid_loss = data['train_loss'], data['val_loss']
        plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
        plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

        # find position of lowest validation loss
        minposs = valid_loss.index(min(valid_loss))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5) # consistent scale
        plt.xlim(0, len(train_loss)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        fig.savefig(os.path.join(self.model_dir,'loss_plot.png'), bbox_inches='tight',dpi=300)


if __name__ == "__main__":
    job = Job()
    writer = job.writer
    job.train()
    job.predict()
