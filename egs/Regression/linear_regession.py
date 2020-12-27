import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error,r2_score
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


def load_dataSet(dataSet,feature_index,target_label):
    lb_maker1 = LabelEncoder()
    lb_maker2 = LabelEncoder()
    dataSet['Temporal Distribution'] = lb_maker1.fit_transform(dataSet['Temporal Distribution'])
    dataSet['Spatial Distribution'] = lb_maker2.fit_transform(dataSet['Spatial Distribution'])
    target = np.array(dataSet[target_label]).reshape((-1,1))
    inputX = np.array(dataSet.loc[:,feature_index])
    trainX,testX,trainY,testY = train_test_split(inputX,target,test_size=0.2,random_state=0)
    return trainX,trainY,testX,testY


class RegressionModel(nn.Module):
    def __init__(self,hidden_num,action_func,in_features):
        super(RegressionModel,self).__init__()
        self.hidden_num = hidden_num
        self.action_func = action_func
        self.linear1 = nn.Linear(in_features=in_features,out_features=hidden_num)
        self.linear2 = nn.Linear(in_features=hidden_num,out_features=1)
        
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias,0.0)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias,0.0)

        
    def forward(self,inputs):
        y1 = self.action_func(self.linear1(inputs))
        # y2 = self.action_func(self.linear2(y1))
        y2 = self.linear2(y1)
        return y2

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

class MyData(Dataset):
    def __init__(self,dataX,dataY):
        self.device = get_device()
        self.dataX = torch.Tensor(dataX).to(self.device)
        self.dataY = torch.Tensor(dataY).to(self.device)

    def __len__(self):
        return len(self.dataY)

    def __getitem__(self,index):
        return self.dataX[index],self.dataY[index]


def evaluate(model,loss_func,dataloader,metrics):
    """Evaluate the model on `num_steps` batches.
    Args:
        model:(torch.nn.Module) the neural network
        loss_func: a function that takes batch_output and batch_lables and compute the loss the batch.
        dataloader:(DataLoader) a torch.utils.data.DataLoader object that fetches data.
        metrics:(dict) a dictionary of functions that compute a metric using the output and labels of each batch.
        num_steps:(int) number of batches to train on,each of size params.batch_size
    """
    model.eval()
    summ = []
    for data in dataloader:
        data_batch,label_batch = data
        data_batch,label_batch = data_batch.cuda(non_blocking=True),label_batch.cuda(non_blocking=True)
        #data_batch,label_batch = Variable(data_batch),Variable(label_batch)
        output_batch = model(data_batch)

        loss = loss_func(output_batch,label_batch).sum(dim=1).mean()

        output_batch = output_batch.data.cpu().numpy()
        label_batch = label_batch.data.cpu().numpy()

        summary_batch = {metric:metrics[metric](output_batch,label_batch) for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k,v) for k,v in metrics_mean.items())
    logging.info("- Eval metrics : "+ metrics_string)
    return metrics_mean



def train(model,optimizer,loss_func,dataloader,metrics):
    """
    Args:
        model:(torch.nn.Module) the neural network
        optimizer:(torch.optim) optimizer for parameters of model
        loss_func: a funtion that takes batch_output and batch_labels and computers the loss for the batch
        dataloader:(DataLoader) a torch.utils.data.DataLoader object that fetchs trainning data

    """
    device = utils.get_device()
    model.to(device)
    model.train()
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i,batch_data in enumerate(dataloader):
            # inputX,_ = batch_data.text
            # inputY = batch_data.label
            inputX,inputY = batch_data
            inputX,inputY = inputX.to(device),inputY.to(device)
            output_batch = model(inputX)
            loss = loss_func(output_batch,inputY).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            if i% 50 == 0:
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = inputY.data.cpu().numpy()

                summary_batch = {metric:metrics[metric](output_batch,labels_batch) for metric in metrics}
                summary_batch['loss'] = loss.item()

                summ.append(summary_batch)
            
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k,v) for k,v in metrics_mean.items())
    logging.info("- Train metrics: "+metrics_string)

def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_func, metrics, epochs, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified

    if restore_file is not None:
        restore_path = os.path.join(model_dir,restore_file+'.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path,model,optimizer)

    best_val_acc = 0.0 #可以替换成f1
    for epoch in range(epochs):
        
        logging.info("Epoch {}/{}".format(epoch+1,epochs))

        train(model,optimizer,loss_func,train_dataloader,metrics)

        val_metircs = evaluate(model,loss_func,val_dataloader,metrics)

        val_acc = val_metircs['r']
        is_best = val_acc >= best_val_acc

        utils.save_checkpoint({'epoch':epoch+1,'state_dict':model.state_dict(),'optim_dict':optimizer.state_dict()},is_best=is_best,checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            best_json_path = os.path.join(model_dir,"val_acc_best_weights.json")
            utils.save_dict_to_json(val_metircs,best_json_path)

        last_json_path = os.path.join(model_dir,"val_acc_last_weights.json")
        utils.save_dict_to_json(val_metircs,last_json_path)

class Job:
    def __init__(self):
        self.log_file = utils.set_logger("train.log")
        self.hidden_num = 100
        self.action_func = nn.Tanh()
        # self.action_func = nn.Sigmoid()
        self.device = get_device()
        # self.action_func = nn.ReLU(inplace=True)
        self.batch_size = 100
        self.epoches = 600
        self.lr = 0.015
        self.feature_index = ['Node Number','Thread Number','T/R','Spatial Distribution','Temporal Distribution']
        self.target_label = 'Channel Waiting Time'
        # self.target_label = 'Input Waiting Time'
        self.xlsfile = "data2.xlsx"
        self.dataSet = pd.read_excel(self.xlsfile)

        self.trainX,self.trainY,self.testX,self.testY = load_dataSet(self.dataSet,self.feature_index,self.target_label)


    def train(self):
        train_data = MyData(self.trainX,self.trainY)
        valid_data = MyData(self.testX,self.testY)

        train_dataloader = DataLoader(dataset=train_data,sampler=RandomSampler(train_data),batch_size=self.batch_size,shuffle=False,num_workers=0,collate_fn=default_collate,drop_last=True)

        valid_dataloader = DataLoader(dataset=valid_data,sampler=RandomSampler(valid_data),batch_size=len(valid_data),shuffle=False,num_workers=0,drop_last=False)

        model = RegressionModel(hidden_num=self.hidden_num,action_func = self.action_func,in_features=len(self.feature_index))
        model.to(device=self.device)
        optim = torch.optim.Adam(model.parameters(), lr = self.lr,betas=(0.9,0.99))
        criterion = torch.nn.MSELoss(reduction="none")
        train_and_evaluate(model,train_dataloader,valid_dataloader,optim,criterion,utils.metrics,self.epoches,model_dir=self.target_label,restore_file=None)
        curr_hyp = {"epochs":self.epoches,"batch_size":self.batch_size,"lr":self.lr,"hidden_num":self.hidden_num}
        utils.save_dict_to_json(curr_hyp,os.path.join(self.target_label,"train_hyp.json"))

    def predict(self):
        valid_data = MyData(self.testX,self.testY)
        valid_dataloader = DataLoader(dataset=valid_data,sampler=RandomSampler(valid_data),batch_size=len(valid_data),shuffle=False,num_workers=0,drop_last=False)
        model = RegressionModel(hidden_num=self.hidden_num,action_func = self.action_func,in_features=len(self.feature_index))
        utils.load_checkpoint(os.path.join(self.target_label,"best.pth.tar"),model)
        model.to(device=self.device)
        model.eval()
        for batch in valid_dataloader:
            inputX,inputY = batch
            inputX = inputX.to(self.device)
            inputY = inputY.to(self.device)
            y_pred = model(inputX).data.cpu().numpy().squeeze()
            y_true = inputY.data.cpu().numpy().squeeze()
            result = pd.DataFrame(data={'y_true':y_true,'y_pred':y_pred},index=range(len(y_pred)))
            result.to_csv("{}_result.csv".format(self.target_label)) 
            print("r2:{},rmse:{}".format(utils.r_score(y_pred,y_true),utils.rmse(y_pred,y_true)))

    def draw_picture(self):
        data = pd.read_csv("{}_result.csv".format(self.target_label))
        fig = plt.gcf()
        sns.regplot(x=data['y_true'],y=data['y_pred'],fit_reg=True)
        plt.xlim(0,data['y_true'].max())
        plt.ylim(0,data['y_true'].max())
        plt.xlabel("Actual {}".format(self.target_label))
        plt.ylabel("Predicted {}".format(self.target_label))
        plt.grid(axis='y') #设置y轴网格线
        fig.savefig(r"{0}/{0}_scatter.png".format(self.target_label), format='png', transparent=True, dpi=300, pad_inches = 0)



if __name__ == "__main__":
    job = Job()
    job.train()
    # job.predict()
    # job.draw_picture()


