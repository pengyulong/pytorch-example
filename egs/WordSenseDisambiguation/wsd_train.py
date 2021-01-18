import torch
from transformers import BertModel, BertTokenizer
import json
import codecs
from tqdm import tqdm
import numpy as np
from stemming.porter2 import stem
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
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("./model")

def load_json(json_file):
    with codecs.open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    data_dict = {}
    for data in data_list:
        data_dict[data['id']] = data
    return data_dict

def extrace_range(string):
    ranges = []
    for data in string.split(','):
        bpos, epos = int(data.split('-')[0]),int(data.split('-')[1])
        ranges.append([bpos,epos])
    return ranges

def load_text_json(text_json):
    dataX,id_list = [],[]
    text_dict = load_json(text_json)
    for key, value in tqdm(text_dict.items()):
        s1, s2 = value['sentence1'], value['sentence2']
        ranges1,ranges2 = [],[]
        if 'start1' in value:
            ranges1.append([int(value['start1']),int(value['end1'])])
            ranges2.append([int(value['start2']),int(value['end2'])])
        else:
            ranges1 = extrace_range(value['ranges1'])
            ranges2 = extrace_range(value['ranges2'])
        id_list.append(key)
        dataX.append([s1,ranges1,s2,ranges2])
    return dataX,id_list

def process_sentence(sentence, ranges, mode='zh'):
    if mode in ['zh','ar']:
        ranges = sorted(ranges,key=lambda x:x[0])
        for i,(start,end) in enumerate(ranges):
            if end+2*i == len(sentence):
                sentence = ','.join([sentence[:start+i*2],sentence[start+i*2:end+i*2]])
            else:
                sentence = ','.join([sentence[:start+i*2],sentence[start+i*2:end+i*2],sentence[end+i*2:]])
            ranges[i][0] += (2*i+1)
            ranges[i][1] += (2*i+1)
    # print("sentence:{},ranges:{}".format(sentence,ranges))
    token_list = []
    for [start, end] in ranges:
        target = sentence[start:end]
        bpos = len(bert_tokenizer.tokenize(sentence[:start]))
        target_token = bert_tokenizer.tokenize(target)
        if target_token[0] == '▁':
            target_token_len = len(target_token) - 1
        else:
            target_token_len = len(target_token)
        for i in range(bpos + 1, bpos + target_token_len + 1):
            token_list.append(i)
    return sentence, token_list


def load_dataSet(text_json, label_json):
    """加载text_json和label_json
    """
    dataX, dataY = [], []
    text_dict, label_dict = load_json(text_json), load_json(label_json)
    for key, value in tqdm(text_dict.items()):
        s1, s2 = value['sentence1'], value['sentence2']
        ranges1, ranges2 = [], []
        if 'start1' in value:
            ranges1.append([int(value['start1']), int(value['end1'])])
            ranges2.append([int(value['start2']), int(value['end2'])])
        else:
            ranges1 = extrace_range(value['ranges1'])
            ranges2 = extrace_range(value['ranges2'])
        # label = label_dict[key]['tag']
        tag = label_dict[key]['tag']
        dataY.append(1 if tag == 'T' else 0)
        dataX.append([s1, ranges1, s2, ranges2])
    return dataX, dataY


def split_dataSet(text_json, label_json):
    inputX, target = load_dataSet(text_json, label_json)
    trainX, testX, trainY, testY = train_test_split(
        inputX, target, test_size=0.2, random_state=0)
    return trainX, trainY, testX, testY


class MyData(Dataset):
    def __init__(self, sentences1,sentences2,ranges1,ranges2,labels):
        self.sentence1 = sentences1
        self.sentence2 = sentences2
        self.ranges1 = ranges1
        self.ranges2 = ranges2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sentence1[index],self.ranges1[index],self.sentence2[index],self.ranges2[index],self.labels[index]

    @classmethod
    def from_list(cls,inputs,target):
        sentence1, sentence2, ranges1, ranges2, labels = [], [], [], [], []
        for dataX, dataY in zip(inputs, target):
            text1, range1, text2, range2 = dataX
            sentence1.append(text1)
            sentence2.append(text2)
            ranges1.append(range1)
            ranges2.append(range2)
            labels.append(dataY)
        return cls(sentence1, sentence2, ranges1, ranges2, labels)


def collate_func(batch):
    setences1,ranges1,sentence2,ranges2,labels = zip(*batch)
    labels = torch.LongTensor(labels)
    return setences1,ranges1,sentence2,ranges2,labels


class WordDisambiguationNet(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, in_features, tokenizer_type, nhead=1, num_layers=1, num_class=2):
        super(WordDisambiguationNet, self).__init__()
        self.num_class = num_class
        self.nhead = nhead
        self.num_layers = num_layers
        self.in_features = in_features
        self.bert_model = bert_model.to(utils.get_device())
        self.bert_tokenizer = bert_tokenizer
        self.tokenizer_type = tokenizer_type
        self.encoder_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.in_features, nhead=self.nhead), num_layers=self.num_layers)
        self.sim = nn.CosineSimilarity(dim=1)
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=2, out_features=self.num_class),
            nn.BatchNorm1d(num_features=self.num_class),
            nn.Sigmoid()
        )
        self.avgpool = nn.AvgPool1d(2)

    def forward(self, sentences1,ranges1,sentences2,ranges2):
        # print("sentences")
        cosine = self._get_similarity(sentences1,ranges1,sentences2,ranges2)
        out1, out2 = cosine.unsqueeze(1), (1-cosine).unsqueeze(1)
        out = torch.cat((out1, out2), dim=1)
        return self.fc_layer(out)

    def _select_embedding(self,sentences,range_list):
        post_sentences, lemma_id_list = [], []
        for i, ranges in enumerate(range_list):
            s, lemma_id = process_sentence(sentences[i],ranges,mode=self.tokenizer_type)
            post_sentences.append(s)
            lemma_id_list.append(lemma_id)
        encoder_inputs = self.bert_tokenizer(post_sentences,return_tensors='pt',padding=True).to(utils.get_device())
        output = self.bert_model(**encoder_inputs)
        lemma_embedings = torch.zeros(len(sentences),self.in_features).to(utils.get_device())
        for i,lemma_id in enumerate(lemma_id_list):
            lemma_embedings[i] = output[0][i,lemma_id,:].mean(dim=0)
        return lemma_embedings.unsqueeze(1)

    def _get_similarity(self,sentences1,ranges1,sentences2,ranges2):
        vec1 = self._select_embedding(sentences1, ranges1)
        vec2 = self._select_embedding(sentences2, ranges2)
        concat = torch.cat((vec1-vec2, vec2-vec1, vec1, vec2),dim=1)
        output = self.encoder_layer(concat)
        output = output.permute(0, 2, 1)
        output = self.avgpool(output)
        cosine = self.sim(output[:, 0, :], output[:, 1, :])
        return cosine

def evaluate(model, loss_func, dataloader, metrics):
    """Evaluate the model on `num_steps` batches.
    Args:
        model:(torch.nn.Module) the neural network
        loss_func: a function that takes batch_output and batch_lables and compute the loss the batch.
        dataloader:(DataLoader) a torch.utils.data.DataLoader object that fetches data.
        metrics:(dict) a dictionary of functions that compute a metric using the output and labels of each batch.
        num_steps:(int) number of batches to train on,each of size params.batch_size
    """
    model.eval()
    summ = []
    device = utils.get_device()
    with torch.no_grad():
        for data in dataloader:
            sentences1, ranges1, sentences2, ranges2, inputY = data
            inputY = inputY.to(device)
            output_batch = model(sentences1, ranges1, sentences2, ranges2)
            loss = loss_func(output_batch, inputY)
            output_batch = output_batch.data.cpu().numpy()
            inputY = inputY.data.cpu().numpy()
            summary_batch = {metric: metrics[metric](
                output_batch, inputY) for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
    # print("summ:{}".format(summ))
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


def train(model, optimizer, loss_func, dataloader, metrics, lr_scheduler):
    """
    Args:
        model:(torch.nn.Module) the neural network
        optimizer:(torch.optim) optimizer for parameters of model
        loss_func: a funtion that takes batch_output and batch_labels and computers the loss for the batch
        dataloader:(DataLoader) a torch.utils.data.DataLoader object that fetchs trainning data

    """
    device = utils.get_device()
    model.train()
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(dataloader)) as t:
        for i, batch_data in enumerate(dataloader):
            # print("batch_data:{}".format(batch_data))
            sentences1, ranges1, sentences2, ranges2, inputY = batch_data
            inputY = inputY.to(device)
            output_batch = model(sentences1, ranges1, sentences2, ranges2)
            loss = loss_func(output_batch, inputY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if i % 50 == 0:
                output_batch = output_batch.data.cpu().numpy()
                inputY = inputY.data.cpu().numpy()
                summary_batch = {metric: metrics[metric](
                    output_batch, inputY) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
        # print("summ:{}".format(summ))
        metrics_mean = {metric: np.mean(
            [x[metric] for x in summ]) for metric in summ[0]}
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                    for k, v in metrics_mean.items())
        logging.info("- Train metrics: "+metrics_string)
    return metrics_mean['loss']


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_func, metrics, epochs, model_dir, lr_scheduler, restore_file=None):
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
    train_loss_list, val_loss_list = [], []
    early_stopping = utils.EarlyStopping(patience=20, verbose=True)
    
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file+'.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_f1 = 0.0  # 可以替换成acc
    for epoch in range(epochs):
        logging.info("lr = {}".format(lr_scheduler.get_last_lr()))
        logging.info("Epoch {}/{}".format(epoch+1, epochs))

        train_loss = train(model, optimizer, loss_func,
                           train_dataloader, metrics, lr_scheduler)

        val_metircs = evaluate(model, loss_func, val_dataloader, metrics)
        # rmse_record.append(val_metircs['rmse'])
        val_loss = val_metircs['loss']
        # loss_result_list.append((train_loss,val_loss))
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        val_f1 = val_metircs['f1']
        is_best = val_f1 >= best_val_f1

        utils.save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(
        ), 'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best accuracy")
            best_val_f1 = val_f1

            best_json_path = os.path.join(
                model_dir, "val_acc_best_weights.json")
            utils.save_dict_to_json(val_metircs, best_json_path)

        last_json_path = os.path.join(model_dir, "val_acc_last_weights.json")
        utils.save_dict_to_json(val_metircs, last_json_path)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping!")
            break
    # return rmse_record
    return {"train_loss": train_loss_list, "val_loss": val_loss_list}


class Job:
    def __init__(self,seed):
        self.log_file = utils.set_logger("./train.log")
        self.device = utils.get_device()
        self.batch_size = 32
        self.epoches = 10
        self.lr = 5e-6
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.train_text_json = r"dataset/training/training.en-en.data"
        self.train_label_json = r"dataset/training/training.en-en.gold"
        self.valid_text_json = r"dataset/dev/multilingual/dev.en-en.data"
        self.valid_label_json = r"dataset/dev/multilingual/dev.en-en.gold"
        self.seed = seed

        self.num_class = 2
        # self.dropout = -0.2
        self.in_features = 768
        self.loss_result = None
        self.warm_ratio = 0.1
        self.model_dir = "End2endXLMRoBertaNet_v2_{}".format(self.seed)
        utils.setup_seed(seed)

    def few_shot_train(self):
        # text_json = r"dataset/trial/crosslingual/trial.en-ru.data"
        # label_json = r"dataset/trial/crosslingual/trial.en-ru.gold"
        # test_text_json = r"dataset/test_few_shot/crosslingual/test.en-ru.data"
        # test_label_json = r"dataset/test_few_shot/crosslingual/test.en-ru.gold"

        text_json = r"dataset/trial/multilingual/trial.zh-zh.data"
        label_json = r"dataset/trial/multilingual/trial.zh-zh.gold"
        test_text_json = r"dataset/test_few_shot/multilingual/test.zh-zh.data"
        test_label_json = r"dataset/test_few_shot/multilingual/test.zh-zh.gold"
        
        tokenizer_type = text_json.split(os.sep)[-1].split('-')[1].split('.')[0]
        device = self.device
        output_dir = text_json.split(os.sep)[-1][0:-5]
        dataX,dataY = load_dataSet(text_json, label_json)
        test_data = MyData.from_list(dataX, dataY)
        test_dataloader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False, num_workers=0, drop_last=False,collate_fn=collate_func)
        model = WordDisambiguationNet(bert_model=bert_model, bert_tokenizer=bert_tokenizer, in_features=768,tokenizer_type=tokenizer_type)
        utils.load_checkpoint(os.path.join(self.model_dir, "best.pth.tar"), model)
        model.to(device=device)
        model.eval()
        avg_cosine0, avg_cosine1 = None, None
        result_list = []
        with torch.no_grad():
            for batch in test_dataloader:
                sentences1, ranges1, sentences2, ranges2, inputY = batch
                inputY = inputY.to(device)
                cosines = model._get_similarity(sentences1,ranges1,sentences2,ranges2)
                cosines = cosines.detach().cpu().numpy().squeeze()
                y_true = inputY.detach().cpu().numpy().squeeze()
                avg_class_0, avg_class_1 = [], []
                for i, label in enumerate(y_true):
                    if label == 1:
                        avg_class_1.append(cosines[i])
                    else:
                        avg_class_0.append(cosines[i])
                avg_cosine0 = np.mean(avg_class_0)
                avg_cosine1 = np.mean(avg_class_1)
                y_pred = []
                for cosine in cosines:
                    if abs(cosine - avg_cosine0) <= abs(cosine - avg_cosine1):
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
                f1 = f1_score(y_true,y_pred)
                acc = accuracy_score(y_true,y_pred)
                print("训练集{}--f1:{},acc:{}".format(text_json.split(os.sep)[-1],f1,acc))
            texts,ids = load_text_json(test_text_json)
            # print("texts:{}".format(texts))
            # print("ids:{}".format(ids))
            for id_name, text in tqdm(zip(ids,texts)):
                # print("text:{}".format(text))
                sentences1,ranges1,sentences2,ranges2 = text
                cosine = model._get_similarity([sentences1],[ranges1],[sentences2],[ranges2])
                cosine = cosine.detach().cpu().numpy().squeeze()
                # print("cosine:{}".format(cosine))
                if abs(cosine - avg_cosine0) <= abs(cosine - avg_cosine1):
                    label = "F"
                else:
                    label = "T"
                result_list.append({"id":id_name,"tag":label})
        with open(test_label_json,"w") as f:
            json.dump(result_list,f)



    def train(self):
        # self.trainX, self.trainY, self.testX, self.testY = split_dataSet(
        #     self.text_json, self.label_json)
        self.trainX,self.trainY = load_dataSet(self.train_text_json,self.train_label_json)
        self.validX,self.validY = load_dataSet(self.valid_text_json,self.valid_label_json)

        train_data = MyData.from_list(self.trainX, self.trainY)
        # print("train_data:{}".format(train_data[0]))
        valid_data = MyData.from_list(self.validX, self.validY)
        train_dataloader = DataLoader(dataset=train_data, sampler=RandomSampler(
            train_data), batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False,collate_fn=collate_func,pin_memory=True)

        valid_dataloader = DataLoader(
            dataset=valid_data, batch_size=(self.batch_size), shuffle=False, num_workers=0,collate_fn=collate_func,drop_last=False)
        model = WordDisambiguationNet(
            bert_model=self.bert_model, bert_tokenizer=self.bert_tokenizer, in_features=self.in_features,tokenizer_type='en')
        model.to(device=self.device)
        XLMRoberta_params = list(map(id,model.bert_model.parameters()))
        base_params = filter(lambda p:id(p) not in XLMRoberta_params,model.parameters())

        optimizer = torch.optim.SGD(
            [
                {"params":model.bert_model.parameters(),"lr":4e-5},
                {"params":base_params},
            ],
            momentum=0.95,weight_decay=0.01,lr=0.001
        )
        # optimizer = torch.optim.AdamW(
        #     optimizer_grouped_parameters, lr=self.lr, eps=1e-8)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warm_ratio*(self.epoches*len(train_dataloader)), num_training_steps=self.epoches*len(train_dataloader))

        criterion = nn.CrossEntropyLoss()
        self.loss_result = train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer,
                                              criterion, utils.classify_metrics, self.epoches, self.model_dir, lr_scheduler, restore_file=None)
        curr_hyp = {"epochs": self.epoches,
                    "batch_size": self.batch_size, "lr": self.lr}
        utils.save_dict_to_json(curr_hyp, os.path.join(
            self.model_dir, "train_hyp.json"))
        df = pd.DataFrame(
            data={'val': self.loss_result['val_loss'], 'train': self.loss_result['train_loss']})
        df.to_csv("{}/loss.csv".format(self.model_dir))

    def evaluate(self):
        text_json = r"dataset/dev/multilingual/dev.en-en.data"
        label_json = r"dataset/dev/multilingual/dev.en-en.gold"
        # text_json = r"dataset/trial/crosslingual/trial.en-zh.data"
        # label_json = r"dataset/trial/crosslingual/trial.en-zh.gold"
        tokenizer_type = text_json.split(os.sep)[-1].split('-')[1].split('.')[0]
        device = self.device
        output_dir = text_json.split(os.sep)[-1][0:-5]
        if os.path.exists(output_dir)==False:
            os.mkdir(output_dir)
        dataX,dataY = load_dataSet(text_json, label_json)
        test_data = MyData.from_list(dataX, dataY)
        test_dataloader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False, num_workers=0, drop_last=False,collate_fn=collate_func)
        model = WordDisambiguationNet(bert_model=bert_model, bert_tokenizer=bert_tokenizer, in_features=768,tokenizer_type=tokenizer_type)
        utils.load_checkpoint(os.path.join(self.model_dir, "best.pth.tar"), model)
        model.to(device=device)
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                sentences1, ranges1, sentences2, ranges2, inputY = batch
                inputY = inputY.to(device)
                output_batch = model(sentences1, ranges1, sentences2, ranges2)
                y_pred = np.argmax(output_batch.detach().cpu().numpy(), axis=1).squeeze()
                y_true = inputY.detach().cpu().numpy().squeeze()
                result = pd.DataFrame(
                    data={'y_true': y_true, 'y_pred': y_pred}, index=range(len(y_pred)))
                # result.to_csv(r"{}/result.csv".format(output_dir))
                print("--测试集{}-{}上的性能指标f1:{},acc:{}".format(text_json.split(os.sep)[1],output_dir,f1_score(y_true,y_pred), accuracy_score(y_true, y_pred)))

   
    def predict(self):
        text_json = r"dataset/test/crosslingual/test.en-ar.data"
        label_json = r"dataset/test/crosslingual/test.en-ar.gold"
        # text_json = r"dataset/test/multilingual/test.zh-zh.data"
        # label_json = r"dataset/test/multilingual/test.zh-zh.gold"
        tokenizer_type = text_json.split(os.sep)[-1].split('-')[1].split('.')[0]
        # test_data, test_id = load_text_json(text_json)
        model = WordDisambiguationNet(bert_model=bert_model,bert_tokenizer=bert_tokenizer,in_features=self.in_features,tokenizer_type=tokenizer_type)
        utils.load_checkpoint(os.path.join(
            self.model_dir, "best.pth.tar"), model)
        model.to(device=self.device)
        model.eval()
        result_list = []
        with torch.no_grad():
            texts,ids = load_text_json(text_json)
            for id_name, text in tqdm(zip(ids,texts)):
                # print("text:{}".format(text))
                (sentence1,ranges1,sentence2,ranges2) = text
                output = model([sentence1],[ranges1],[sentence2],[ranges2])
                y_pred = np.argmax(output.detach().cpu().numpy(),axis=1).squeeze()
                label = "T" if y_pred == 1 else "F"
                result_list.append({"id":id_name,"tag":label})
        with open(label_json,"w") as f:
            json.dump(result_list,f)

    def draw_picture(self):
        plt.figure()
        if self.loss_result is None:
            df = pd.read_csv(
                "{}/loss.csv".format(self.model_dir))
            epochs = range(len(df))
            val_loss = df['val']
            train_loss = df['train']
        else:
            epochs = range(len(self.loss_result['val_loss']))
            val_loss = self.loss_result['val_loss']
            train_loss = self.loss_result['train_loss']
        # val_loss = []
        plt.plot(epochs, train_loss, label='train')
        plt.plot(epochs, val_loss, label='valid')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(axis='both', linestyle='-.')  # 设置网格线
        plt.savefig(r"{0}/{1}_loss.png".format(self.model_dir, self.mode),
                    format='png', transparent=True, dpi=300, pad_inches=0)


if __name__ == "__main__":
    for seed in [2021,43,26,48,78]:
        job = Job(seed=seed)
        job.train()
    # job.train()
    # job.evaluate()
    # # job.few_shot_train()
    # # job.predict()

