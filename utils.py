import random
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,roc_auc_score
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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_currTime():
    curr = time.localtime()
    currStr = "{}{}{:02d}{:02d}{:02d}{:02d}".format(curr.tm_year,curr.tm_mon,curr.tm_mday,curr.tm_hour,curr.tm_min,curr.tm_sec)

    return currStr


def get_device():
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


metrics = {'f1':f1,'acc':acc,'recall':recall,'precision':precision}

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
        d = {k: float(v) for k, v in d.items()}
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
    checkpoint = torch.load(checkpoint)
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