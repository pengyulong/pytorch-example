import torch
from transformers import BertModel, BertTokenizer
import json
import codecs
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data import Dataset
import numpy as np
import utils
from torch.utils.data.dataloader import default_collate
import logging
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from nltk.tokenize import word_tokenize
from torchsummary import summary

bert_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base',is_split_into_words=True)
bert_model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

utils.setup_seed(2020)

def process_sentence(sentence,start,end):
    target = sentence[start:end]
    prev = sentence[:start]
    target_token = bert_tokenizer.tokenize(target)
    if target_token[0] == '▁':
        target_token_len = len(target_token) - 1
    else:
        target_token_len = len(target_token)
    # target_token_len = len(bert_tokenizer.tokenize(target))
    npos = len(bert_tokenizer.tokenize(prev))
    token_list = list(range(npos+1,npos+target_token_len+1))
    return token_list

def load_json(json_file):
    with codecs.open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list


def load_text_json(text_json):
    dataX,id_list = [],[]
    text_list = load_json(text_json)
    for text in text_list:
        id_list.append(text['id'])
        if 'ranges1' in text:
            pos_range1 = text['ranges1'].split(',')[0]
            pos_range2 = text['ranges2'].split(',')[0]
            start1, end1 = int(pos_range1.split('-')[0]),int(pos_range1.split('-')[1])
            start2, end2 = int(pos_range2.split('-')[0]),int(pos_range2.split('-')[1])
        else:
            start1,end1 = int(text['start1']),int(text['end1'])
            start2,end2 = int(text['start2']),int(text['end2'])
        text1,text2 = text['sentence1'],text['sentence2']
        dataX.append([text1,start1,end1,text2,start2,end2])
    return dataX,id_list

def load_dataSet(text_json, label_json):
    """加载text_json和label_json
    """
    dataX, dataY = [], []
    text_list, label_list = load_json(text_json), load_json(label_json)
    for text, label in tqdm(zip(text_list, label_list)):
        text1, text2 = text['sentence1'], text['sentence2']
        if 'ranges1' in text:
            pos_range1 = text['ranges1'].split(',')[0]
            pos_range2 = text['ranges2'].split(',')[0]
            start1, end1 = int(pos_range1.split('-')[0]),int(pos_range1.split('-')[1])
            start2, end2 = int(pos_range2.split('-')[0]),int(pos_range2.split('-')[1])
        else:
            start1,end1 = int(text['start1']),int(text['end1'])
            start2,end2 = int(text['start2']),int(text['end2'])
        tag, word = label['tag'], text['lemma']
        dataY.append(1 if tag == 'T' else 0)
        dataX.append([text1, text2, start1, end1, start2, end2])
    return dataX, dataY

class MyData(Dataset):
    def __init__(self, sentences1,sentences2,starts1,ends1,starts2,ends2,labels):
        self.sentence1 = sentences1
        self.sentence2 = sentences2
        self.starts1 = starts1
        self.ends1 = ends1
        self.starts2 = starts2
        self.ends2 = ends2
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sentence1[index], self.starts1[index], self.ends1[index],self.sentence2[index], self.starts2[index],self.ends2[index], self.labels[index]

    @classmethod
    def from_list(cls,inputs,target):
        sentence1, sentence2, starts1,ends1,starts2,ends2,labels = [], [], [], [], [], [], []
        for dataX, dataY in zip(inputs, target):
            text1, text2, start1, end1, start2, end2 = dataX
            sentence1.append(text1)
            sentence2.append(text2)
            starts1.append(start1)
            ends1.append(end1)
            starts2.append(start2)
            ends2.append(end2)
            labels.append(dataY)
        return cls(sentence1, sentence2, starts1, ends1, starts2, ends2, labels)


class WordDisambiguationNet(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, in_features, nhead=1, num_layers=1, num_class=2):
        super(WordDisambiguationNet, self).__init__()
        self.num_class = num_class
        self.nhead = nhead
        self.num_layers = num_layers
        self.in_features = in_features
        self.bert_model = bert_model.to(utils.get_device())
        self.bert_tokenizer = bert_tokenizer
        self.encoder_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.in_features, nhead=self.nhead), num_layers=self.num_layers)
        self.sim = nn.CosineSimilarity(dim=1)
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.num_class),
            nn.BatchNorm1d(num_features=self.num_class),
            nn.Sigmoid()
        )
        self.avgpool = nn.AvgPool1d(2)

    def forward(self, sentences1,start1,end1,sentences2,start2,end2):
        vec1 = self._select_embedding(sentences1,start1,end1)
        vec2 = self._select_embedding(sentences2,start2,end2)
        concat = torch.cat((vec1-vec2, vec2-vec1, vec1, vec2),dim=1)
        output = self.encoder_layer(concat)
        output = output.permute(0, 2, 1)
        output = self.avgpool(output)

        cosine = self.sim(output[:, 0, :], output[:, 1, :])
        out1, out2 = cosine.unsqueeze(1), (1-cosine).unsqueeze(1)
        out = torch.cat((out1, out2), dim=1)
        return self.fc_layer(out)

    def _select_embedding(self,sentences,starts,ends):
        encoder_inputs = self.bert_tokenizer(sentences,return_tensors='pt',padding=True).to(utils.get_device())
        output = self.bert_model(**encoder_inputs)
        lemma_embedings = torch.zeros(len(sentences),self.in_features).to(utils.get_device())
        for i,(start, end) in enumerate(zip(starts,ends)):
            lemma_ids = process_sentence(sentences[i],start,end)
            lemma_embedings[i] = output[0][i,lemma_ids,:].mean(dim=0)
        return lemma_embedings.unsqueeze(1)


def batch_pred():
    model_dir = r"End2endXLMRoBertaNet_nochop"
    device = utils.get_device()
    # text_json = r"data/development/dev.zh-zh.data"
    # label_json = r"data/development/dev.zh-zh.gold"
    text_json = r"data/trial/crosslingual/trial.en-zh.data"
    label_json = r"data/trial/crosslingual/trial.en-zh.gold"

    # text_json = r"data/trial/multilingual/trial.zh-zh.data"
    # label_json = r"data/trial/multilingual/trial.zh-zh.gold"
    output_dir = text_json.split(os.sep)[-1][0:-5]
    if os.path.exists(output_dir)==False:
        os.mkdir(output_dir)
    dataX,dataY = load_dataSet(text_json, label_json)
    test_data = MyData.from_list(dataX, dataY)
    test_dataloader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False, num_workers=0, drop_last=False)
    model = WordDisambiguationNet(bert_model=bert_model, bert_tokenizer=bert_tokenizer, in_features=768)
    utils.load_checkpoint(os.path.join(model_dir, "best.pth.tar"), model)
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            sentences1, starts1, ends1, sentences2, starts2, ends2, inputY = batch
            inputY = inputY.to(device)
            output_batch = model(sentences1, starts1, ends1, sentences2, starts2, ends2)
            y_pred = np.argmax(output_batch.detach().cpu().numpy(), axis=1).squeeze()
            y_true = inputY.detach().cpu().numpy().squeeze()
            result = pd.DataFrame(
                data={'y_true': y_true, 'y_pred': y_pred}, index=range(len(y_pred)))
            result.to_csv(r"{}/result.csv".format(output_dir))
            print("--测试集{}-{}上的性能指标f1:{},acc:{}".format(text_json.split(os.sep)[1],output_dir,f1_score(y_true,y_pred), accuracy_score(y_true, y_pred)))

def single_file_predict(filename,outfile,model_dir):
    model = WordDisambiguationNet(bert_model=bert_model, bert_tokenizer=bert_tokenizer, in_features=768)
    device = utils.get_device()
    # device = torch.device("cpu")
    utils.load_checkpoint(os.path.join(model_dir,"best.pth.tar"),model)
    model.to(device)
    model.eval()
    result_list = []
    with torch.no_grad():
        texts,ids = load_text_json(filename)
        for id_name, text in tqdm(zip(ids,texts)):
            # print("text:{}".format(text))
            sentence1,start1,end1,sentence2,start2,end2 = text
            output = model([sentence1],[start1],[end1],[sentence2],[start2],[end2])
            # print("sentence1:{},sentence")
            y_pred = np.argmax(output.detach().cpu().numpy(),axis=1).squeeze()
            label = "T" if y_pred == 1 else "F"
            result_list.append({"id":id_name,"tag":label})
    with open(outfile,"w") as f:
        json.dump(result_list,f)




if __name__ == "__main__":
    test_json = r"test/crosslingual/test.en-ar.data"
    output_json = r"test/crosslingual/test.en-ar.gold"
    model_dir = r"End2endXLMRoBertaNet_nochop"
    single_file_predict(test_json,output_json,model_dir)
    


