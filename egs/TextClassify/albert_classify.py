import torch
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import pandas as pd
from utils import text_filter
import utils
from tqdm import tqdm
import os
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch.nn as nn
import pickle
from transformers import get_linear_schedule_with_warmup


utils.setup_seed(2020)

class AlbertClassfier(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, bert_config, num_class):
        super(AlbertClassfier, self).__init__()
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.dropout = nn.Dropout(0.2)
        self.num_class = num_class
        self.fc1 = nn.Linear(
            bert_config.hidden_size, bert_config.hidden_size)
        self.fc2 = nn.Linear(bert_config.hidden_size, self.num_class)

    def forward(self, inputs):
        encoder_inputs = self.bert_tokenizer(inputs,return_tensors='pt',padding=True)
        encoder_inputs = encoder_inputs.to(utils.get_device())
        # 句向量 [batch_size,hidden_size]
        bert_out = self.bert_model(**encoder_inputs)[1]
        bert_out = self.dropout(bert_out)
        bert_out = self.fc1(bert_out)
        bert_out = self.dropout(bert_out)
        bert_out = self.fc2(bert_out)  # [batch_size,num_class]
        return bert_out


class SentimentData(Dataset):
    def __init__(self, data, label):
        # print("label:{}".format(label))
        self.device = utils.get_device()
        self.data = data
        self.label = torch.LongTensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    @classmethod
    def from_txt(cls, csvfile):
        """
            
        """
        dataSet = pd.read_csv(csvfile,sep='\t',header=None)
        dataSet.columns = ['content','label']
        dataSet['content'] = dataSet['content'].apply(text_filter)

        dataX, dataY = [], []
        for (_, row) in tqdm(dataSet.iterrows()):
            content = text_filter(row['content'])
            label = int(row['label']) # [1,0,-1,-2] ->[3,2,1,0]
            dataX.append(content)
            dataY.append(label)
        # sorted_result = sorted(enumerate(dataX),key=lambda x:len(x[1]))
        # dataX = [e[1] for e in sorted_result]
        # index = [e[0] for e in sorted_result]
        # dataY = [dataY[i] for i in index]
        return cls(dataX, dataY)

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


class Job:
    def __init__(self):
        self.device = get_device()
        self.batch_size = 64
        self.epoches = 10
        self.lr = 2e-5
        self.num_class = 10
        self.sent_class = r"TilteClassify"
        self.train_file = r"data2/train.txt"
        self.valid_file = r"data2/test.txt"
        self.train_pkl_file = r"data2/train.pkl"
        self.valid_pkl_file = r"data2/test.pkl"
        self.max_length = 500
        self.warmup_ratio = 0.1

        self.pretrained_name = "hfl/chinese-roberta-wwm-ext-large"
        self.albert_tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_name)
        self.albert_model = BertModel.from_pretrained(self.pretrained_name)
        self.albert_config = BertConfig.from_pretrained(self.pretrained_name)

        if os.path.isfile(self.train_pkl_file):
            with open(self.train_pkl_file,"rb") as f:
                self.train_dataset = pickle.load(f)
        else:
            self.train_dataset = SentimentData.from_txt(self.train_file)
            with open(self.train_pkl_file,"wb") as f:
                pickle.dump(self.train_dataset,f)
        if os.path.isfile(self.valid_pkl_file):
            with open(self.valid_pkl_file,"rb") as f:
                self.valid_dataset = pickle.load(f)
        else:
            with open(self.valid_pkl_file,"wb") as f:
                self.valid_dataset = SentimentData.from_txt(self.valid_file)
                pickle.dump(self.valid_dataset,f)

        self.model_dir = "./{}".format(self.sent_class)
        if os.path.exists(self.model_dir) == False:
            os.mkdir(self.model_dir)
        self.log_file = utils.set_logger(
            os.path.join(self.model_dir, "train.log"))

    def train(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, sampler=RandomSampler(
            self.train_dataset), batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=default_collate, drop_last=False)
        valid_dataloader = DataLoader(
            dataset=self.valid_dataset, batch_size=self.batch_size//2, shuffle=False, num_workers=0, drop_last=False)
        model = AlbertClassfier(
            self.albert_model, self.albert_tokenizer, self.albert_config, self.num_class)
        # model.to(device=self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= self.warmup_ratio*self.epoches*len(train_dataloader), num_training_steps=self.epoches*len(train_dataloader))
        utils.train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer, criterion,
                                 utils.metrics, self.epoches, self.model_dir,lr_scheduler,restore_file=None)
        curr_hyp = {"epochs": self.epoches,
                    "batch_size": self.batch_size, "lr": self.lr}
        utils.save_dict_to_json(curr_hyp, os.path.join(
            self.model_dir, "train_hyp.json"))

    def predict(self):
        valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=len(
            self.valid_dataset), shuffle=False, num_workers=0, drop_last=False)
        model = AlbertClassfier(
            self.albert_model, self.bert_tokenizer, self.albert_config, self.num_class)
        utils.load_checkpoint(os.path.join(
            self.model_dir, "best.pth.tar"), model)
        # model.to(device=self.device)
        model.eval()
        y_preds, y_trues = [], []
        for data in valid_dataloader:
            inputX, inputY = data
            # inputX = inputX.to(self.device)
            inputY = inputY.to(self.device)
            y_pred = np.argmax(
                model(inputX).data.cpu().numpy(), axis=1).squeeze()
            y_true = inputY.data.cpu().numpy().squeeze()
            y_preds.extend(y_pred.tolist())
            y_trues.extend(y_true.tolist())
        result = pd.DataFrame(
            data={'y_true': y_trues, 'y_pred': y_preds}, index=range(len(y_preds)))
        result.to_csv(r"{}/result.csv".format(self.model_dir))
        evaluate_model(y_preds, y_trues)


if __name__ == "__main__":
    job = Job()
    job.train()
    job.predict()
