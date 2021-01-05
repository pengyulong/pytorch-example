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
from utils import get_device,train_and_evaluate,save_dict_to_json,save_result_dict_list,load_checkpoint,setup_seed,classify_metrics,set_logger
from model import BertSequenceClassfier
import matplotlib.pyplot as plt

setup_seed(2020)

class SentimentData(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = torch.LongTensor(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    @classmethod
    def from_csv(cls, csvfile, tokenizer, sent_class, max_length=510):
        """
        sent_class由20个类别组成,分别是location_traffic_convenience,location_distance_from_business_district,
        location_easy_to_find,service_wait_time,service_waiters_attitude,service_parking_convenience,
        service_serving_speed,price_level,price_cost_effective,price_discount,environment_decoration,
        environment_noise,environment_space,environment_cleaness,dish_portion,dish_taste,dish_look,
        dish_recommendation,others_overall_experience,others_willing_to_consume_again'
        """
        dataSet = pd.read_csv(csvfile, index_col=0)
        # dataSet['content'].apply(text_filter,inplace=True)
        dataSet['content'] = dataSet['content'].apply(text_filter)
        dataX, dataY = [], []
        for (_, row) in tqdm(dataSet.iterrows()):
            content = text_filter(row['content'])
            tokens = tokenizer.tokenize(content)
            if len(tokens) >= max_length:
                continue
            label = int(row[sent_class]) + 2 # [1,0,-1,-2] ->[3,2,1,0]
            dataX.append(content)
            dataY.append(label)
        return cls(dataX, dataY)


def save_prob_file(probs,y_true,filename):
    test_prob=pd.DataFrame(probs)
    num_outputs = len(probs[0])
    test_id = [i for i in range(len(probs))]
    test_prob.columns=["class_prob_%s"%i for i in range(1,num_outputs+1)]
    test_prob['id']=list(test_id)
    test_prob['y_true'] = y_true
    test_prob.to_csv(filename,index=None)
    return True

class Job:
    def __init__(self,sent_class,pool_type):
        self.device = get_device()
        self.batch_size = 128
        self.epoches = 1
        self.lr = 2e-5
        self.num_class = 4
        self.sent_class = sent_class
        self.train_file = r"data/train.csv"
        self.valid_file = r"data/valid.csv"
        self.max_length = 100
        self.warmup_ratio = 0.1
        self.pool_type = pool_type # avg,max,cls 
        self.model_dir = "./{}/{}".format(self.sent_class,self.pool_type)
        if os.path.exists(self.model_dir) == False:
            os.makedirs(self.model_dir)
        self.train_pkl_file = os.path.join(self.sent_class,"train.pkl")
        self.valid_pkl_file = os.path.join(self.sent_class,"valid.pkl")
        # self.pretrained_name = "hfl/chinese-roberta-wwm-ext-large"
        self.pretrained_name = "hfl/chinese-roberta-wwm-ext"
        # self.pretrained_name = "voidful/albert_chinese_small"
        self.albert_tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_name)
        self.albert_model = BertModel.from_pretrained(self.pretrained_name)
        self.albert_config = BertConfig.from_pretrained(self.pretrained_name)
        if os.path.isfile(self.train_pkl_file):
            with open(self.train_pkl_file,"rb") as f:
                self.train_dataset = pickle.load(f)
        else:
            self.train_dataset = SentimentData.from_csv(self.train_file,self.albert_tokenizer,self.sent_class,self.max_length)
            with open(self.train_pkl_file,"wb") as f:
                pickle.dump(self.train_dataset,f)
        if os.path.isfile(self.valid_pkl_file):
            with open(self.valid_pkl_file,"rb") as f:
                self.valid_dataset = pickle.load(f)
        else:
            self.valid_dataset = SentimentData.from_csv(self.valid_file,self.albert_tokenizer,self.sent_class,self.max_length)
            with open(self.valid_pkl_file,"wb") as f:
                pickle.dump(self.valid_dataset,f)
        self.log_file = set_logger(
            os.path.join(self.model_dir, "train.log"))

    def train(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, sampler=RandomSampler(
            self.train_dataset), batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=default_collate, drop_last=False)
        valid_dataloader = DataLoader(
            dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        model = BertSequenceClassfier(self.albert_model,self.albert_tokenizer,self.albert_config,num_class=self.num_class,pool_type=self.pool_type)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= self.warmup_ratio*self.epoches*len(train_dataloader), num_training_steps=self.epoches*len(train_dataloader))


        train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer, criterion,classify_metrics, self.epoches, self.model_dir,lr_scheduler,restore_file=None)
        curr_hyp = {"epochs": self.epoches,
                    "batch_size": self.batch_size, "lr": self.lr, "pool_type":self.pool_type}
        save_dict_to_json(curr_hyp, os.path.join(
            self.model_dir, "train_hyp.json"))


    def predict(self):
        valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=len(
            self.valid_dataset), shuffle=False, num_workers=0, drop_last=False)
        model = BertSequenceClassfier(self.albert_model,self.albert_tokenizer,self.albert_config,num_class=self.num_class,pool_type=self.pool_type)
        load_checkpoint(os.path.join(
            self.model_dir, "best.pth.tar"), model)
        model.to(device=self.device)
        model.eval()
        y_preds, y_trues = [], []
        with torch.no_grad():
            for data in valid_dataloader:
                inputX, inputY = data
                inputY = inputY.to(self.device)
                y_pred = model(inputX).detach().cpu().numpy().squeeze()
                y_true = inputY.detach().cpu().numpy().squeeze()
                y_preds.extend(y_pred.tolist())
                y_trues.extend(y_true.tolist())

        save_prob_file(y_preds,y_trues,os.path.join(self.sent_class,"result_{}.csv".format(self.pool_type)))


if __name__ == "__main__":
    job = Job("location_traffic_convenience","max")
    # job.train()
    job.predict()
