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
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY

    def __len__(self):
        return len(self.dataY)

    def __getitem__(self, index):
        return self.dataX[index],self.dataY[index]

    @classmethod
    def from_table(cls, textfile):
        """"""
        dataSet = pd.read_csv(textfile,sep='\t',header=None)
        dataSet.columns = ['content','label']
        dataSet['content'] = dataSet['content'].apply(text_filter)
        dataX, dataY = [], []
        for (_, row) in tqdm(dataSet.iterrows()):
            content = text_filter(row['content'])
            label = int(row['label']) # [1,0,-1,-2] ->[3,2,1,0]
            dataX.append(content)
            dataY.append(label)
        return cls(dataX, dataY)


class Job:
    def __init__(self):
        self.device = get_device()
        self.batch_size = 64
        self.epoches = 5
        self.lr = 2e-5
        self.num_class = 10
        self.sent_class = "TilteClassify"
        self.train_file = r"data2/train.txt"
        self.valid_file = r"data2/test.txt"
        self.train_pkl_file = r"data2/train.pkl"
        self.valid_pkl_file = r"data2/test.pkl"
        self.pool_type = "avg"
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
            self.train_dataset = SentimentData.from_table(self.train_file)
            with open(self.train_pkl_file,"wb") as f:
                pickle.dump(self.train_dataset,f)
        if os.path.isfile(self.valid_pkl_file):
            with open(self.valid_pkl_file,"rb") as f:
                self.valid_dataset = pickle.load(f)
        else:
            with open(self.valid_pkl_file,"wb") as f:
                self.valid_dataset = SentimentData.from_table(self.valid_file)
                pickle.dump(self.valid_dataset,f)

        self.model_dir = "./{}".format(self.sent_class)
        if os.path.exists(self.model_dir) == False:
            os.mkdir(self.model_dir)
        self.log_file = set_logger(
            os.path.join(self.model_dir, "train.log"))

    def train(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, sampler=RandomSampler(
            self.train_dataset), batch_size=self.batch_size, shuffle=False, num_workers=0,drop_last=False)

        valid_dataloader = DataLoader(
            dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        
        model = BertSequenceClassfier(self.albert_model,self.albert_tokenizer,self.albert_config,num_class=self.num_class,pool_type=self.pool_type)    


        model.to(device=self.device)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.lr, eps = 1e-8)

        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= self.warmup_ratio*(self.epoches*len(train_dataloader)), num_training_steps=self.epoches*len(train_dataloader))
        criterion = nn.CrossEntropyLoss()
        self.loss_result = train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer, criterion,
                                 classify_metrics, self.epoches, self.model_dir, lr_scheduler, restore_file=None)
        curr_hyp = {"epochs": self.epoches,
                    "batch_size": self.batch_size, "lr": self.lr, "pool_type":self.pool_type}
        save_dict_to_json(curr_hyp, os.path.join(
            self.model_dir, "train_hyp.json"))
        save_result_dict_list(self.loss_result,os.path.join(self.model_dir,"loss.csv"))

        
    def predict(self):
        valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0, drop_last=False)
        model = BertSequenceClassfier(self.albert_model,self.albert_tokenizer,self.albert_config,num_class=self.num_class,pool_type=self.pool_type) 
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
    # job.train()
    job.predict()
    job.plot_loss()
