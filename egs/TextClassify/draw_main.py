import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from sent_classify import Job
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from utils import text_filter
from transformers import BertTokenizer

matplotlib.use("Agg")

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
            content = row['content']
            # content = text_filter(row['content'])
            tokens = tokenizer.tokenize(content)
            if len(tokens) >= max_length:
                continue
            label = int(row[sent_class]) + 2 # [1,0,-1,-2] ->[3,2,1,0]
            dataX.append(content)
            dataY.append(label)
        return cls(dataX, dataY)


def load_loss(log_file):
    train_loss,valid_loss = [],[]
    with open(log_file,'r',encoding='utf8') as f:
        for line in f.readlines():
            data = line.split(':')
            if " - Train metrics" in line:
                f1 = float(data[6].split(';')[0])
                acc = float(data[7].split(';')[0])
                recall = float(data[8].split(';')[0])
                precision = float(data[9].split(';')[0])
                loss = float(data[10].split(';')[0])
                train_loss.append({"f1":f1,"acc":acc,"recall":recall,"precision":precision,"loss":loss})
            if " - Eval metrics" in line:
                f1 = float(data[6].split(';')[0])
                acc = float(data[7].split(';')[0])
                recall = float(data[8].split(';')[0])
                precision = float(data[9].split(';')[0])
                loss = float(data[10].split(';')[0])
                valid_loss.append({"f1":f1,"acc":acc,"recall":recall,"precision":precision,"loss":loss})
    
    train_df = pd.DataFrame(train_loss)
    valid_df = pd.DataFrame(valid_loss)

    return train_df, valid_df


def draw_figure(train_data,valid_data,model_dir,variable):
    fig = plt.figure()
    plt.plot(range(1,len(train_data[variable])+1),train_data[variable],label="Training {}".format(variable))
    plt.plot(range(1,len(valid_data[variable])+1),valid_data[variable],label="Validation {}".format(variable))
    # if variable == 'loss':
    #     minposs = .index(min(valid))+1 
    #     plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    plt.ylim(0,1)
    plt.xlim(0,len(train_data[variable])+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(model_dir,"{}.png".format(variable)),bbox_inches='tight',dpi=300)

def dataSet_info(csvfile,tokenizer):
    dataSet = pd.read_csv(csvfile, index_col=0)
    dataSet['content'] = dataSet['content'].apply(text_filter)
    lengths = [len(tokenizer.tokenize(text)) for text in dataSet['content']]
    fig = plt.figure()
    plt.hist(lengths,bins=50,normed=1,facecolor='blue',edgecolor='black')
    plt.xlabel("文本长度")
    plt.ylabel("比例")
    plt.title("文本长度分布直方图")
    fig.savefig("length_ratio.png",bbox_inches='tight',dpi=300)



if __name__ == "__main__":
    csvfile = r'data/train.csv'
    pretrained_name = "hfl/chinese-roberta-wwm-ext"
    albert_tokenizer = BertTokenizer.from_pretrained(pretrained_name)

    # sent_classes = ['location_traffic_convenience']
    # # sent_classes = ["location_distance_from_business_district","location_easy_to_find","price_cost_effective","location_traffic_convenience","location_distance_from_business_district","others_willing_to_consume_again","service_parking_convenience"]
    # model_names = ['avg','max','CLS','out','cnn','rnn']

    # for sent_class in sent_classes:
    #     for pool_type in model_names:
    #         model_dir = os.path.join(sent_class,pool_type)
    #         logfile = os.path.join(model_dir,"train.log")
    #         if os.path.exists(logfile) == False:
    #             print("开始训练{}情感分类,模型为:{}".format(sent_class,pool_type))
    #             job = Job(sent_class,pool_type)
    #             job.train()
    #             job.predict()
    #         train_data,valid_data = load_loss(logfile)
    #         print("训练{}情感分类完毕,开始画图".format(sent_class,pool_type))
    #         for var in ['f1','acc','recall','precision','loss']:
    #             draw_figure(train_data,valid_data,model_dir,var)


                


