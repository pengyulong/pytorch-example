import torch
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import pandas as pd
from utils import text_filter
from tqdm import tqdm
import os
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import pickle
from transformers import get_linear_schedule_with_warmup
from utils import get_device,train_and_evaluate,save_dict_to_json,save_result_dict_list,load_checkpoint,setup_seed,classify_metrics,set_logger,split_dataSet
from model import BertSequenceClassfier,BertEncoderClassfier
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
setup_seed(2020)

def loadDataSet(data_dir):
    if data_dir == 'weibo_moods':
        dataSet = pd.read_csv(os.path.join(data_dir,'data.csv'))
    if data_dir == "holtel_sent":
        dataSet = pd.read_csv(os.path.join(data_dir,'data.txt'),sep='    ',header=None)
        dataSet.columns = ['label','review']
    trainX, trainY, testX, testY = split_dataSet(dataSet['label'],dataSet['review'])
    return trainX, trainY, testX, testY


class SentimentData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.LongTensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    @classmethod
    def from_list(cls, contexts, labels, tokenizer, max_length=510):
        """ 通过list对象的contexts和 labels初始化数据集
        """
        dataX, dataY = [], []
        for text,label in tqdm(zip(contexts,labels)):
            content = text_filter(text)
            tokens = tokenizer.tokenize(content)
            if len(tokens) >= max_length:
                continue
            label = int(label)
            dataX.append(content)
            dataY.append(label)
        return cls(dataX, dataY)


class Job:
    def __init__(self,data_dir, pool_type):
        self.device = get_device()
        self.batch_size = 16
        self.epoches = 5
        self.lr = 2e-5
        self.data_dir = data_dir
        self.max_length = 510
        self.warmup_ratio = 0.1
        self.pool_type = pool_type # BiGRU, Att_BiGRU, DPCNN, max
        self.trainX, self.trainY, self.validX, self.validY = loadDataSet(self.data_dir)
        self.num_class = len(set(self.validY))
        self.model_dir = "./{}/{}".format(self.data_dir,self.pool_type)
        if os.path.exists(self.model_dir) == False:
            os.makedirs(self.model_dir)
        self.train_pkl_file = os.path.join(self.data_dir,"train.pkl")
        self.valid_pkl_file = os.path.join(self.data_dir,"valid.pkl")
        self.pretrained_name = "hfl/chinese-roberta-wwm-ext"
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_name)
        self.bert_model = BertModel.from_pretrained(self.pretrained_name)
        self.bert_config = BertConfig.from_pretrained(self.pretrained_name)
        if os.path.isfile(self.train_pkl_file):
            with open(self.train_pkl_file,"rb") as f:
                self.train_dataset = pickle.load(f)
        else:
            self.train_dataset = SentimentData.from_list(self.trainX,self.trainY,self.bert_tokenizer)
            with open(self.train_pkl_file,"wb") as f:
                pickle.dump(self.train_dataset,f)
        if os.path.isfile(self.valid_pkl_file):
            with open(self.valid_pkl_file,"rb") as f:
                self.valid_dataset = pickle.load(f)
        else:
            self.valid_dataset = SentimentData.from_list(self.validX,self.validY,self.bert_tokenizer)
            with open(self.valid_pkl_file,"wb") as f:
                pickle.dump(self.valid_dataset,f)
        self.log_file = set_logger(r"{}/train.log".format(self.model_dir))

    def draw_figure(self,variable):
        fig = plt.figure()
        plt.plot(range(1,len(self.train_df[variable])+1),self.train_df[variable],label="Training {}".format(variable))
        plt.plot(range(1,len(self.valid_df[variable])+1),self.valid_df[variable],label="Validation {}".format(variable))
        # if variable == 'loss':
        #     minposs = .index(min(valid))+1 
        #     plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        plt.ylim(0,1)
        plt.xlim(1,len(self.train_df[variable])+1)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.model_dir,"{}.png".format(variable)),bbox_inches='tight',dpi=300)


    def train(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, sampler=RandomSampler(
            self.train_dataset), batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=default_collate, pin_memory=True, drop_last=False)
        valid_dataloader = DataLoader(
            dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)

        if self.pool_type in ['Att_BiGRU','BiGRU','DPCNN']:
            model = BertEncoderClassfier(self.bert_model, self.bert_tokenizer, self.num_class, self.bert_config, max_length = self.max_length, encoder_type = self.pool_type, dropout=0.2)
        else:
            model = BertSequenceClassfier(self.bert_model,self.bert_tokenizer,self.bert_config,num_class=self.num_class,pool_type=self.pool_type)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= self.warmup_ratio*self.epoches*len(train_dataloader), num_training_steps=self.epoches*len(train_dataloader))
        self.train_loss,self.valid_loss = train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer, criterion,classify_metrics, self.epoches, self.model_dir,lr_scheduler,restore_file=None)
        curr_hyp = {"epochs": self.epoches,
                    "batch_size": self.batch_size, "lr": self.lr, "pool_type":self.pool_type}
        save_dict_to_json(curr_hyp, os.path.join(
            self.model_dir, "train_hyp.json"))
        self.train_df = pd.DataFrame(data=self.train_loss)
        self.valid_df = pd.DataFrame(data=self.valid_loss)
        self.train_df.to_excel(os.path.join(self.model_dir,"train_loss.xlsx"))
        self.valid_df.to_excel(os.path.join(self.model_dir,"valid_loss.xlsx"))


    def predict(self):
        valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)
        if self.pool_type in ['Att_BiGRU','BiGRU','DPCNN']:
            model = BertEncoderClassfier(self.bert_model, self.bert_tokenizer, self.num_class, self.bert_config, max_length = self.max_length, encoder_type = self.pool_type, dropout=0.2)
        else:
            model = BertSequenceClassfier(self.bert_model,self.bert_tokenizer,self.bert_config,num_class=self.num_class,pool_type=self.pool_type)
        load_checkpoint(os.path.join(
            self.model_dir, "best.pth.tar"), model)
        model.to(self.device)
        model.eval()
        y_preds, y_trues = [], []
        for data in tqdm(valid_dataloader):
            inputX, inputY = data
            # inputX = inputX.to(self.device)
            inputY = inputY.to(self.device)
            y_pred = np.argmax(
                model(inputX).detach().cpu().numpy(), axis=1).squeeze()
            y_true = inputY.detach().cpu().numpy().squeeze()
            y_preds.extend(y_pred.tolist())
            y_trues.extend(y_true.tolist())
        result = pd.DataFrame(
            data={'y_true': y_trues, 'y_pred': y_preds}, index=range(len(y_preds)))
        result.to_csv(r"{}/result.csv".format(self.model_dir))


if __name__ == "__main__":
    model_dirs = ['holtel_sent','weibo_moods']
    pool_types = ['max','BiGRU','AttBiGRU','DPCNN']
    for model_dir in model_dirs:
        for pool_type in pool_types:
            job = Job(model_dir,pool_type)
            job.train()
            job.predict()