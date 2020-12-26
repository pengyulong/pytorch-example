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
from utils import get_device,train_and_evaluate,save_dict_to_json,save_result_dict_list,load_checkpoint,setup_seed
from model import BertSequenceClassfier
import matplotlib.pyplot as plt


setup_seed(2020)

class SentimentData(Dataset):
    def __init__(self, encoder_inputs, labels, tokenizer):
        self.albert_tokenizer = tokenizer
        self.encoder_inputs = encoder_inputs
        self.label = labels

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        item = {key:torch.tensor(val[index]) for key,val in (self.encoder_inputs).items()}
        item['label'] = torch.tensor(self.label[index])
        return item

    @classmethod
    def from_csv(cls, csvfile, sent_class, tokenizer, max_length=500):
        """
            sent_class由20个类别组成,分别是location_traffic_convenience,location_distance_from_business_district,
            location_easy_to_find,service_wait_time,service_waiters_attitude,service_parking_convenience,
            service_serving_speed,price_level,price_cost_effective,price_discount,environment_decoration,
            environment_noise,environment_space,environment_cleaness,dish_portion,dish_taste,dish_look,
            dish_recommendation,others_overall_experience,others_willing_to_consume_again'
        """
        dataSet = pd.read_csv(csvfile,sep='\t',header=None)
        dataSet.columns = ['text','label']
        dataX, dataY = [], []
        for i, (index, row) in tqdm(enumerate(dataSet.iterrows())):
            content = filter(row['text'])
            length = len(content)
            if length >= max_length:
                continue
            label = int(row[sent_class]) + 2 # [1,0,-1,-2] ->[3,2,1,0]
            dataX.append(content)
            dataY.append(label)
        encoder_inputs = tokenizer(dataX,return_tensors='pt',padding=True)
        return cls(encoder_inputs, dataY, tokenizer)


class Job:
    def __init__(self):
        self.device = get_device()
        self.batch_size = 32
        self.epoches = 100
        self.lr = 0.00001
        self.num_class = 4
        self.sent_class = "location_traffic_convenience"
        self.train_file = r"data/train.csv"
        self.valid_file = r"data/valid.csv"
        self.train_pkl_file = r"data/train.pkl"
        self.valid_pkl_file = r"data/valid.pkl"
        self.pool_type = "avg"
        self.max_length = 500

        self.pretrained_name = "voidful/albert_chinese_small"
        self.albert_tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_name)
        self.albert_model = BertModel.from_pretrained(self.pretrained_name)
        self.albert_config = BertConfig.from_pretrained(self.pretrained_name)

        if os.path.isfile(self.train_pkl_file):
            with open(self.train_pkl_file,"rb") as f:
                self.train_dataset = pickle.load(f)
        else:
            self.train_dataset = SentimentData.from_csv(self.train_file, self.sent_class, self.albert_tokenizer)
            with open(self.train_pkl_file,"wb") as f:
                pickle.dump(self.train_dataset,f)
        if os.path.isfile(self.valid_pkl_file):
            with open(self.valid_pkl_file,"rb") as f:
                self.valid_dataset = pickle.load(f)
        else:
            with open(self.valid_pkl_file,"wb") as f:
                self.valid_dataset = SentimentData.from_csv(self.valid_file, self.sent_class, self.albert_tokenizer)
                pickle.dump(self.valid_dataset,f)

        self.model_dir = "./{}".format(self.sent_class)
        if os.path.exists(self.model_dir) == False:
            os.mkdir(self.model_dir)
        self.log_file = utils.set_logger(
            os.path.join(self.model_dir, "train.log"))

    def train(self):
        train_dataloader = DataLoader(dataset=self.train_dataset, sampler=RandomSampler(
            self.train_dataset), batch_size=self.batch_size, shuffle=False, num_workers=0,drop_last=False)

        valid_dataloader = DataLoader(
            dataset=self.valid_dataset, batch_size=self.batch_size//2, shuffle=False, num_workers=0, drop_last=False)
        
        model = BertSequenceClassfier(self.albert_model,self.albert_tokenizer,self.albert_config,num_class=self.num_class,pool_type=self.pool_type)    


        # model.to(device=self.device)
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

        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= 500, num_training_steps=self.epoches*len(train_dataloader))
        criterion = nn.CrossEntropyLoss()
        self.loss_result = utils.train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer, criterion,
                                 utils.metrics, self.epoches, self.model_dir, lr_scheduler, restore_file=None)
        curr_hyp = {"epochs": self.epoches,
                    "batch_size": self.batch_size, "lr": self.lr}
        save_dict_to_json(curr_hyp, os.path.join(
            self.model_dir, "train_hyp.json"))
        save_result_dict_list(self.loss_result,os.path.join(self.model_dir,"loss.csv"))

        
    def predict(self):
        valid_dataloader = DataLoader(dataset=self.valid_dataset, batch_size=len(
            self.valid_dataset), shuffle=False, num_workers=0, drop_last=False)
        model = BertSequenceClassfier(self.albert_model,self.albert_tokenizer,self.albert_config,num_class=self.num_class,pool_type=self.pool_type) 
        load_checkpoint(os.path.join(
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
    job.train()
    job.predict()
    job.plot_loss()
