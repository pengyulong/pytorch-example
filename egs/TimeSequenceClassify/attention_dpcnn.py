import torch.functional as F
from sklearn import preprocessing
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.optim import lr_scheduler
import pandas as pd
from torch.utils.data.dataloader import default_collate
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import pandas as pd
from tensorboardX import SummaryWriter
from utils import load_checkpoint, classify_metrics, setup_seed, set_logger, get_device, train_and_evaluate, save_dict_to_json, read_sequence_data
from model import SequenceClassify
from config import load_args

setup_seed(2020)  # 设置随机数种子,保证结果可复现

class SequenceData(Dataset):
    def __init__(self, dataX1, dataX2, dataX3, dataX4, dataX5, dataX6, dataX7, dataY):
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

    def __getitem__(self, index):
        return self.dataX1[index], self.dataX2[index], self.dataX3[index], self.dataX4[index], self.dataX5[index], self.dataX6[index], self.dataX7[index], self.dataY[index]

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
    device = utils.get_device()
    summ = []
    with torch.no_grad():
        for data in dataloader:
            inputX1,inputX2,inputX3,inputX4,inputX5,inputX6,inputX7,label_batch = data
            label_batch = label_batch.to(device)
            inputX1 = inputX1.to(device)
            inputX2 = inputX2.to(device)
            inputX3 = inputX3.to(device)
            inputX4 = inputX4.to(device)
            inputX5 = inputX5.to(device)
            inputX6 = inputX6.to(device)
            inputX7 = inputX7.to(device)
            output_batch = model(inputX1,inputX2,inputX3,inputX4,inputX5,inputX6,inputX7)

            loss = loss_func(output_batch,label_batch)

        
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
            inputX1,inputX2,inputX3,inputX4,inputX5,inputX6,inputX7,inputY = batch_data
            inputY = inputY.to(device)
            inputX1 = inputX1.to(device)
            inputX2 = inputX2.to(device)
            inputX3 = inputX3.to(device)
            inputX4 = inputX4.to(device)
            inputX5 = inputX5.to(device)
            inputX6 = inputX6.to(device)
            inputX7 = inputX7.to(device)
            output_batch = model(inputX1,inputX2,inputX3,inputX4,inputX5,inputX6,inputX7)
            loss = loss_func(output_batch,inputY)
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
    return {"f1":metrics_mean['f1'],"loss":metrics_mean['loss'],"acc":metrics_mean['acc']}


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_func, metrics, epochs, model_dir,restore_file=None):
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
    scheduler = lr_scheduler.StepLR(optimizer,step_size=40,gamma=0.1)

    train_loss, valid_loss = [], []

    best_val_f1 = 0.0 
    for epoch in range(epochs):
        logging.info("Epoch {}/{}".format(epoch+1,epochs))

        train_metrics = train(model,optimizer,loss_func,train_dataloader,metrics)
        train_loss.append(train_metrics)

        val_metircs = evaluate(model,loss_func,val_dataloader,metrics)
        valid_loss.append({"f1":val_metircs['f1'],"acc":val_metircs['acc'],"loss":val_metircs['loss']})
        scheduler.step()
        logging.info("lr:{}".format(scheduler.get_last_lr()))

        val_f1 = val_metircs['f1']
        is_best = val_f1 >= best_val_f1
        writer.add_scalar("val_loss",val_metircs['loss'],epoch)
        writer.add_scalar("val_acc",val_metircs['acc'],epoch)
        writer.add_scalar("val_f1",val_metircs['f1'],epoch)
        utils.save_checkpoint({'epoch':epoch+1,'state_dict':model.state_dict(),'optim_dict':optimizer.state_dict()},is_best=is_best,checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_f1 = val_f1

            best_json_path = os.path.join(model_dir,"val_f1_best_weights.json")
            utils.save_dict_to_json(val_metircs,best_json_path)

        last_json_path = os.path.join(model_dir,"val_f1_last_weights.json")
        utils.save_dict_to_json(val_metircs,last_json_path)
    return valid_loss,train_loss

def save_confusion_matrix(result_data, savetxt):
    y_true, y_pred = result_data['y_true'], result_data['y_pred']
    confusion_matrix = classification_report(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    with open(savetxt, "w", encoding='utf-8') as f:
        print("f1-macro:{:.3f},f1-micro:{:.3f},acc:{:.3f},precision:{:.3f},recall:{:.3f}".format(
            f1_macro, f1_micro, acc, precision, recall), file=f)
        print(confusion_matrix, file=f)
    return True


def draw_figure(train_data, valid_data, pngfile, variable):
    train = train_data[variable].tolist()
    valid = valid_data[variable].tolist()

    fig = plt.figure()
    plt.plot(range(1, len(train)+1), train,
             label="Training {}".format(variable))
    plt.plot(range(1, len(valid)+1), valid,
             label="Validation {}".format(variable))
    if variable == 'loss':
        minposs = valid.index(min(valid))+1
        plt.axvline(minposs, linestyle='--', color='r',
                    label='Early Stopping Checkpoint')
    plt.ylim(0, 1)
    plt.xlim(0, len(train)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(pngfile, bbox_inches='tight', dpi=300)


class Job:
    def __init__(self):
        args = load_args()
        self.device = get_device()
        self.batch_size = args.batch_size
        self.epoches = args.epoches
        self.lr = args.lr
        self.trainX1, self.trainY, self.testX1, self.testY, seq_length1 = read_sequence_data(
            r"train_data/主机电流样本.csv")
        self.trainX2, self.trainY, self.testX2, self.testY, seq_length2 = read_sequence_data(
            r"train_data/负压样本.csv")
        self.trainX3, self.trainY, self.testX3, self.testY, seq_length3 = read_sequence_data(
            r"train_data/料浆样本.csv")
        self.trainX4, self.trainY, self.testX4, self.testY, seq_length4 = read_sequence_data(
            r"train_data/喂煤样本.csv")
        self.trainX5, self.trainY, self.testX5, self.testY, seq_length5 = read_sequence_data(
            r"train_data/窑头温度样本.csv")
        self.trainX6, self.trainY, self.testX6, self.testY, seq_length6 = read_sequence_data(
            r"train_data/窑尾温度样本.csv")
        self.trainX7, self.trainY, self.testX7, self.testY, seq_length7 = get_data(
            r"train_data/一次风样本.csv")
        self.seq_lengths = [seq_length1, seq_length2, seq_length3,
                            seq_length4, seq_length5, seq_length6, seq_length7]
        self.num_class = args.num_class
        self.out_channels = args.filter_num  # [75,150,169,207,209,129]
        self.hidden_num = args.hidden_num
        self.loss_type = args.loss_type  # CROSS, SMOOTH
        self.model_dir = "./result_{}/{}".format(
            self.out_channels, self.loss_type)
        if os.path.exists(self.model_dir) == False:
            os.makedirs(self.model_dir)
        self.log_file = utils.set_logger(
            r"./result_{}/{}/train.log".format(self.out_channels, self.loss_type))

        self.writer = SummaryWriter(
            logdir=os.path.join(self.model_dir, "runs"))

    def train(self):
        train_data = SequenceData(self.trainX1, self.trainX2, self.trainX3,
                                  self.trainX4, self.trainX5, self.trainX6, self.trainX7, self.trainY)
        valid_data = SequenceData(self.testX1, self.testX2, self.testX3,
                                  self.testX4, self.testX5, self.testX6, self.testX7, self.testY)

        train_dataloader = DataLoader(dataset=train_data, sampler=RandomSampler(
            train_data), batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=default_collate, drop_last=False)

        valid_dataloader = DataLoader(
            dataset=valid_data, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        model = SequenceClassify(out_channels=self.out_channels, num_class=self.num_class,
                                 hidden_num=self.hidden_num, seq_lengths=self.seq_lengths)
        model.to(device=self.device)
        optim = torch.optim.AdamW(
            model.parameters(), lr=self.lr, betas=(0.9, 0.99))
        if self.loss_type == 'SMOOTH':
            criterion = utils.LabelSmoothingCrossEntropy()
        if self.loss_type == 'CROSS':
            criterion = nn.CrossEntropyLoss()

        valid_loss, train_loss = train_and_evaluate(model, train_dataloader, valid_dataloader, optim,
                                                    criterion, classify_metrics, self.epoches, model_dir=self.model_dir, restore_file=None)
        curr_hyp = {"epochs": self.epoches, "batch_size": self.batch_size, "lr": self.lr,
                    "hidden_num": self.hidden_num, "out_channels": self.out_channels}
        utils.save_dict_to_json(curr_hyp, os.path.join(
            self.model_dir, "train_hyp.json"))
        valid_df = pd.DataFrame(valid_loss)
        train_df = pd.DataFrame(train_loss)
        valid_df.to_excel(os.path.join(self.model_dir, "valid_loss.xlsx"))
        train_df.to_excel(os.path.join(self.model_dir, "train_loss.xlsx"))

    def predict(self):
        valid_data = SequenceData(self.testX1, self.testX2, self.testX3,
                                  self.testX4, self.testX5, self.testX6, self.testX7, self.testY)
        valid_dataloader = DataLoader(dataset=valid_data, batch_size=len(
            valid_data), shuffle=False, num_workers=0, drop_last=False)

        model = SequenceClassify(out_channels=self.out_channels, num_class=self.num_class,
                                 hidden_num=self.hidden_num, seq_lengths=self.seq_lengths)
        load_checkpoint(os.path.join(self.model_dir, "best.pth.tar"), model)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(valid_dataloader):
                inputX1, inputX2, inputX3, inputX4, inputX5, inputX6, inputX7, label_batch = batch
                y_true = label_batch
                y_pred = np.argmax(model(inputX1, inputX2, inputX3, inputX4,
                                         inputX5, inputX6, inputX7).data.cpu().numpy(), axis=1).squeeze()
                result = pd.DataFrame(
                    data={'y_true': y_true, 'y_pred': y_pred}, index=range(len(y_pred)))
                result.to_csv(r"{}/result.csv".format(self.model_dir))

    def plot_loss(self):
        train_data = pd.read_excel(os.path.join(
            self.model_dir, "train_loss.xlsx"))
        valid_data = pd.read_excel(os.path.join(
            self.model_dir, "valid_loss.xlsx"))
        result_data = pd.read_csv(os.path.join(self.model_dir, "result.csv"))
        for variable in ['f1', 'loss', 'acc']:
            pngfile = os.path.join(
                self.model_dir, "{}_descent.png".format(variable))
            draw_figure(train_data, valid_data, pngfile, variable)
            savetext = os.path.join(self.model_dir, "结果指标.txt")
            save_confusion_matrix(result_data, savetext)


if __name__ == "__main__":
    job = Job()
    writer = job.writer
    job.train()
    job.predict()
    job.plot_loss()
