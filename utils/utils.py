import random
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import numpy as np
import json
import shutil
import logging
import jieba
import codecs
import re
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def split_dataSet(inputX, target, test_size=0.2):
    trainX, testX, trainY, testY = train_test_split(
        inputX, target, test_size=test_size, random_state=0)
    return trainX, trainY, testX, testY

def read_sequence_data(csvfile):
    dataSet = pd.read_csv(csvfile,index_col=0,header=None)
    dataX = np.array(dataSet.iloc[:,:-1])
    seq_length = dataX.shape[1]
    dataX = normalize(dataX,axis=1,norm='max')
    dataY = np.array(dataSet.iloc[:,-1]-1)
    trainX,trainY,testX,testY = split_dataSet(dataX,dataY)
    return trainX,trainY,testX,testY,seq_length

def get_device():
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'): 
        super(LabelSmoothingCrossEntropy,self).__init__()
        self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': 
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':  
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

def loss_func(outputs,labels):
    """
    compute the entropy loss given outpus and lables.

    Args:
        outpus:(Variable) dimension batch_size*n_class output of the model.
        labels:(Variable) dimension batch_size*1,where element is a value in [0,1,2,3,4,5]

    Returns:
        loss:(Variable) cross entropy loss for all images in the batch.
    """
    critical = LabelSmoothingCrossEntropy()
    return critical(outputs,labels)

def f1(y_pred,y_true):
    y_pred = np.argmax(y_pred,axis=1)
    # y_true = np.argmax(y_true,axis=1)
    return f1_score(y_pred=y_pred,y_true=y_true,average='macro')

def acc(y_pred,y_true):
    y_pred = np.argmax(y_pred,axis=1)
    # y_true = np.argmax(y_true,axis=1)
    return accuracy_score(y_true,y_pred)

def recall(y_pred,y_true):
    y_pred = np.argmax(y_pred,axis=1)
    # y_true = np.argmax(y_true,axis=1)
    return recall_score(y_true=y_true,y_pred=y_pred,average='macro')

def precision(y_pred,y_true):
    y_pred = np.argmax(y_pred,axis=1)
    # y_true = np.argmax(y_true,axis=1)
    return precision_score(y_pred=y_pred,y_true=y_true,average='macro')

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func


    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.val_loss_min = val_loss

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


classify_metrics = {'f1':f1,'acc':acc,'recall':recall,'precision':precision}


def r_score(y_pred,y_true):
    ans = r2_score(y_true=y_true,y_pred=y_pred)
    return ans if ans < 0 else np.sqrt(ans)

def rmse(y_pred,y_true):
    return np.sqrt(mean_squared_error(y_true=y_true,y_pred=y_pred))

def mae(y_pred,y_true):
    return abs(y_true - y_pred).mean()

def rae(y_pred,y_true):
    return np.sqrt(abs(y_true - y_pred).sum()/abs(y_true - y_true.mean()).sum())

def rrse(y_pred,y_true):
    return np.sqrt(((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum())


regression_metrics = {'r':r_score,'rmse':rmse,'mae':mae,'rae':rae,'rrse':rrse}

class RunningAverage():
    """
    A simple class that maintains the running average of a quantity.

    Example:
    `
        loss_avg = RunningAverage()
        loss_avg.update(2)
        loss_avg.update(4)

        print(loss_avg()) # (2+4)/2
    `
    """
    def __init__(self):
        self.steps = 0
        self.total = 0.0

    def update(self,value):
        self.total += value
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)    

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint,map_location=get_device())
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def save_result_dict_list(result_dict_list,csvfile):
    assert csvfile.endswith('.csv')
    if os.path.exists(csvfile.split(os.sep)[0:-1]) is False:
        os.makedirs(csvfile.split(os.sep)[0:-1])
    data = pd.DataFrame(data=result_dict_list)
    data.to_csv(csvfile)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_func, metrics, epochs, model_dir,lr_scheduler=None, restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_func: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        epoch: (int) a number indicate train epochs
        model_dir: (string) directory containing config, weights and log
        lr_scheduler: (torch.optime) lr_scheduler for learning rate 
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file+'.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, optimizer)

    train_loss_list, val_loss_list = [], []
    early_stopping = EarlyStopping(patience=20,verbose=True)
    val_losses = []

    best_val_f1 = 0.0  # 可以替换成其他评测指标,acc,precision,recall等
    for epoch in range(epochs):

        logging.info("Epoch {}/{}".format(epoch+1, epochs))
        train_loss,train_metrics = train(model, optimizer, loss_func, train_dataloader, metrics, lr_scheduler)
        val_loss,val_metrics = evaluate(model, loss_func, val_dataloader, metrics)
        train_loss_list.extend(train_loss)
        val_loss_list.extend(val_loss)
        
        val_losses.append(val_metrics['loss'])
        val_f1 = val_metrics['f1']
        is_best = val_f1 >= best_val_f1

        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(
        ), 'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best f1-macro")
            best_val_f1 = val_f1

            best_json_path = os.path.join(
                model_dir, "val_f1_best_weights.json")
            save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(model_dir, "val_f1_last_weights.json")
        save_dict_to_json(val_metrics, last_json_path)

        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            logging.info("Early stopping!")
            break

    return train_loss_list,val_loss_list


def train(model, optimizer, loss_func, dataloader, metrics, lr_scheduler=None):
    """
    Args:
        model:(torch.nn.Module) the neural network
        optimizer:(torch.optim) optimizer for parameters of model
        loss_func: a funtion that takes batch_output and batch_labels and computers the loss for the batch
        dataloader:(DataLoader) a torch.utils.data.DataLoader object that fetchs trainning data

    """
    device = get_device()
    model.to(device)
    model.train()
    summ = []
    loss_avg = RunningAverage()
    if lr_scheduler is not None:
        logging.info("lr = {}".format(lr_scheduler.get_last_lr()))
    with tqdm(total=len(dataloader)) as t:
        for step, data in enumerate(dataloader):
            data_batch, label_batch = data
            label_batch = label_batch.to(device)
            output_batch = model(data_batch)
            loss = loss_func(output_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if step % 200 == 0:
                output_batch = output_batch.detach().cpu().numpy()
                label_batch = label_batch.detach().cpu().numpy()
                summary_batch = {metric: metrics[metric](output_batch, label_batch) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: "+metrics_string)

    return summ,metrics_mean


def evaluate(model, loss_func, dataloader, metrics):
    """Evaluate the model on `num_steps` batches.
    Args:
        model:(torch.nn.Module) the neural network
        loss_func: a function that takes batch_output and batch_lables and compute the loss the batch.
        dataloader:(DataLoader) a torch.utils.data.DataLoader object that fetches data.
        metrics:(dict) a dictionary of functions that compute a metric using the output and labels of each batch.
        num_steps:(int) number of batches to train on,each of size params.batch_size
    """
    model.eval()
    device = get_device()
    summ = []
    with torch.no_grad():
        for data in dataloader:
            data_batch, label_batch = data
            label_batch = label_batch.to(device)
            output_batch = model(data_batch)
            loss = loss_func(output_batch, label_batch)
            output_batch = output_batch.detach().cpu().numpy()
            label_batch = label_batch.detach().cpu().numpy()

            summary_batch = {metric: metrics[metric](
                output_batch, label_batch) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return summ,metrics_mean
