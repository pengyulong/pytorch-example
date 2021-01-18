import os
from utils import get_device,classify_metrics
import pandas as pd
import torch.nn as nn
import torch
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
from utils import text_filter
import json,codecs
albert_tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

def load_json(json_file):
    with codecs.open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list

def evaluate_model(y_true,y_pred):
    result = {}
    for key,value in classify_metrics.items():
        result[key] = value(y_pred,y_true)
    return result

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

def load_sample_file(csvfile,max_length,sent_classes):
    dataSet = pd.read_csv(csvfile, index_col=0)
    dataSet['content'] = dataSet['content'].apply(text_filter)
    result = {}
    for sent_class in sent_classes:
        result[sent_class] = {}
    for (_, row) in tqdm(dataSet.iterrows()):
        content = text_filter(row['content'])
        tokens = albert_tokenizer.tokenize(content)
        if len(tokens) >= max_length:
            continue
        for sent_class in sent_classes:
            # label = int(row[sent_class]) + 2 # [1,0,-1,-2] ->[3,2,1,0]
            label = row[sent_class]
            if label in result[sent_class]:
                result[sent_class][label] += 1
            else:
                result[sent_class][label] = 1
    return result

class Job:
    def __init__(self):
        # self.device = get_device()
        self.batch_size = 8
        self.epoches = 5
        self.lr = 2e-5
        self.num_class = 4
        self.max_length = 500
        self.warmup_ratio = 0.1
        # self.pool_type = pool_type # avg,max,cls,cnn,rnn,out 
        self.train_file = r'data/train.csv'
        self.valid_file = r'data/valid.csv'
        self.sent_classes = ["location_distance_from_business_district","location_easy_to_find","price_cost_effective","location_traffic_convenience","others_willing_to_consume_again","service_parking_convenience"]
        self.test_length = 13113

    def stacking_predict2(self):
        model = nn.Softmax(dim=1)
        for sent_class in self.sent_classes:
            y_true = None
            result = {}
            ans = np.zeros((self.test_length,4))
            for pool_type in ['avg','max','CLS','cnn','rnn','out']:
                csvfile = os.path.join(sent_class,"result_{}.csv".format(pool_type))
                dataSet = pd.read_csv(csvfile)
                y_true = dataSet['y_true'].tolist() # 真实值
                prob = model(torch.from_numpy(dataSet.iloc[:,0:4].values))
                print("prob's shape:{}".format(prob.shape))
                print("sent_class:{},csv_file:{}".format(sent_class,csvfile))
                y_pred = prob.detach().numpy().tolist()
                ans += prob.detach().numpy()
                result[pool_type] = evaluate_model(y_true,y_pred)
            # stacking_pred = np.argmax(ans,axis=1).tolist()
            result['stack'] = evaluate_model(y_true,ans.tolist())
            data = pd.DataFrame(data=result)
            data.to_excel("./{}_result.xlsx".format(sent_class))

    def stacking_predict(self):
        model = nn.Softmax(dim=1)
        for sent_class in self.sent_classes:
            y_true = None
            result = {}
            ans = np.zeros((self.test_length,4))
            for pool_type in ['avg','max','CLS','cnn','rnn','out']:
                json_file = r"{}/{}/val_f1_best_weights.json".format(sent_class,pool_type)
                json_data = load_json(json_file)
                print(json_data)
                result[pool_type] = json_data
            # stacking_pred = np.argmax(ans,axis=1).tolist()
            # result['stack'] = evaluate_model(y_true,ans.tolist())
            data = pd.DataFrame(data=result)
            data.to_excel("./{}_result.xlsx".format(sent_class))


    def static_samples(self,max_length=500):
        train_result = load_sample_file(self.train_file,max_length,self.sent_classes)
        valid_result = load_sample_file(self.valid_file,max_length,self.sent_classes)
        train_df = pd.DataFrame(train_result)
        valid_df = pd.DataFrame(valid_result)
        train_df.to_excel("train_sample.xlsx")
        valid_df.to_excel("valid_sample.xlsx")


        
        



if __name__ == "__main__":
    job = Job()
    job.stacking_predict()
    # job.static_samples()

