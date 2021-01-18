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
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader,RandomSampler,SequentialSampler)
from torch.utils.data import Dataset
import numpy as np
import utils
from torch.utils.data.dataloader import default_collate
import logging,os
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("./model")


def process_sentence(sentence, target, window_size, start, end):
    prev = sentence[:start]
    if sentence[end] == 's':  # 如果是复数形式的化,字符串后移一位
        end += 1
    post = sentence[end:]
    token_list = []
    prev_list = tokenizer.tokenize(prev)
    post_list = tokenizer.tokenize(post)
    if window_size > 0:
        token_list.extend(prev_list[max(0, len(prev_list)-window_size):])
        token_list.append(target)
        token_list.extend(post_list[:min(window_size, len(post_list))])
    else:
        token_list.extend(prev_list)
        token_list.append(target)
        token_list.extend(post_list)
    return " ".join(token_list)


def load_json(json_file):
    with codecs.open(json_file, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    return data_list


def load_dataSet(text_json, label_json, window_size):
    """加载text_json和label_json
    """
    # print("window_size:{}".format(window_size))
    dataX, dataY = [], []
    text_list, label_list = load_json(text_json), load_json(label_json)
    for text, label in tqdm(zip(text_list, label_list)):
        text1, text2 = text['sentence1'], text['sentence2']
        start1, end1 = int(text['start1']), int(text['end1'])
        start2, end2 = int(text['start2']), int(text['end2'])
        tag, word = label['tag'], text['lemma']
        dataY.append(1 if tag == 'T' else 0)
        if window_size > 0:
            text1 = process_sentence(text1, word, window_size, start1, end1)
            text2 = process_sentence(text2, word, window_size, start2, end2)
            dataX.append([text1, text2, word])
        else:
            text1 = process_sentence(text1, word, -1, start1, end1)
            text2 = process_sentence(text2, word, -1, start2, end2)
            dataX.append([text1, text2, word])
    return dataX, dataY


def split_dataSet(text_json, label_json, window_size):
    inputX, target = load_dataSet(text_json, label_json, window_size)
    trainX, testX, trainY, testY = train_test_split(
        inputX, target, test_size=0.2, random_state=0)
    return trainX, trainY, testX, testY



def collect_func(batch):
    encoder1,encoder2,pos1_ids,pos2_ids,labels = zip(*batch)
    def zip_dict(encs):
        new_dict = {}
        for key in encs[0].keys():
            vals = []
            for enc in encs:
                vals.append(enc[key].unsqueeze(0))
            new_dict[key] = torch.cat(vals,dim=0)
        return new_dict    
    pos1_ids = [pos_id for pos_id in pos1_ids]
    pos2_ids = [pos_id for pos_id in pos2_ids]
    labels = torch.LongTensor([a.numpy().tolist() for a in labels])
    
    return zip_dict(encoder1),zip_dict(encoder2),pos1_ids,pos2_ids,labels


class MyData(Dataset):
    def __init__(self, encoder_inputs1, encoder_inputs2, pos1_ids, pos2_ids, labels):
        self.encoder_inputs1 = encoder_inputs1
        self.encoder_inputs2 = encoder_inputs2
        self.pos1_ids = pos1_ids
        self.pos2_ids = pos2_ids
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        encoder_input1 = {key:val[index] for key,val in self.encoder_inputs1.items()}
        encoder_input2 = {key:val[index] for key,val in self.encoder_inputs2.items()}
        return encoder_input1,encoder_input2,self.pos1_ids[index],self.pos2_ids[index],self.labels[index]
        
    @classmethod
    def from_list(cls, inputs, target):
        sentence1, sentence2, token_ids, labels = [], [], [], []
        pos1_ids, pos2_ids = [], []
        for dataX, dataY in zip(inputs, target):
            s1, s2, word = dataX
            token_ids = tokenizer.encode(word)[1:-1]
            sentence1.append(s1)
            sentence2.append(s2)
            labels.append(dataY)
            sent1_id = tokenizer.encode(s1)
            sent2_id = tokenizer.encode(s2)
            pos1_ids.append([sent1_id.index(token_id) for token_id in token_ids])
            pos2_ids.append([sent2_id.index(token_id) for token_id in token_ids])
        # print("pos1_ids:{},type of pos1_ids:{}".format(pos1_ids,type(pos1_ids)))
        # print("pos2_ids:{},type of pos2_ids:{}".format(pos2_ids,type(pos2_ids)))
        encoder_input1 = tokenizer(sentence1,return_tensors='pt', padding=True)
        encoder_input2 = tokenizer(sentence2,return_tensors='pt', padding=True)
        return cls(encoder_input1,encoder_input2,pos1_ids,pos2_ids,labels)


class WordDisambiguationNet(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, in_features, nhead=1, num_layers=1, num_class=2, dropout=0.5):
        super(WordDisambiguationNet, self).__init__()
        self.num_class = num_class
        self.nhead = nhead
        self.num_layers = num_layers
        self.in_features = in_features
        self.bert_tokenizer = bert_tokenizer
        self.device = utils.get_device()
        self.bert_model = bert_model.to(self.device)
        self.encoder_layer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=self.in_features, nhead=self.nhead), num_layers=self.num_layers)
        self.sim = nn.CosineSimilarity(dim=1)
        self.dropout = dropout
        if self.dropout < 0:
            self.fc_layer = nn.Sequential(
                nn.Linear(in_features=2, out_features=self.num_class),
                nn.ReLU(inplace=True)
                # nn.Softmax(dim=1)
            )
        else:
            self.fc_layer = nn.Sequential(
                nn.Linear(in_features=2, out_features=self.num_class),
                nn.ReLU(inplace=True),
                # nn.Softmax(dim=1),
                nn.Dropout(p=self.dropout)
            )

    def forward(self,encoder_inputs1,encoder_inputs2,pos1_ids,pos2_ids):
        vec1 = self._select_embedding(encoder_inputs1,pos1_ids)
        vec2 = self._select_embedding(encoder_inputs2,pos2_ids)
        concat = torch.cat((vec1-vec2, vec1+vec2), dim=1)
        output = self.encoder_layer(concat)
        cosine = self.sim(output[:, 0, :], output[:, 1, :])
        out1, out2 = cosine.unsqueeze(1), (1-cosine).unsqueeze(1)
        out = torch.cat((out1, out2), dim=1)
        return self.fc_layer(out)

    def _select_embedding(self, encoder_inputs, pos_ids):
        # output = self.bert_model(**encoder_inputs)

        output = self.bert_model(encoder_inputs['input_ids'].to(self.device),encoder_inputs['attention_mask'].to(self.device))
        token_embedding = torch.zeros(len(pos_ids), self.in_features).to(self.device)
        for i, slice_ids in enumerate(pos_ids):
            token_embedding[i] = output[0][i, slice_ids, :].mean(dim=0)
        return token_embedding.unsqueeze(1)



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
    for data in dataloader:
        encoder_inputs1,encoder_inputs2,pos1_ids,pos2_ids,inputY = data
        inputY = inputY.to(device)
        output_batch = model(encoder_inputs1,encoder_inputs2,pos1_ids,pos2_ids)

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
            encoder_inputs1,encoder_inputs2,pos1_ids,pos2_ids,inputY = batch_data
            inputY = inputY.to(device)
            output_batch = model(encoder_inputs1,encoder_inputs2,pos1_ids,pos2_ids)
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


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_func, metrics, epochs, model_dir, lr_scheduler,restore_file=None):
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
    early_stopping = utils.EarlyStopping(patience=20,verbose=True)

    logging.info("lr = {}".format(lr_scheduler.get_last_lr()))
    if restore_file is not None:
        restore_path = os.path.join(model_dir, restore_file+'.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_f1 = 0.0  # 可以替换成acc
    for epoch in range(epochs):

        logging.info("Epoch {}/{}".format(epoch+1, epochs))

        train_loss = train(model, optimizer, loss_func,
                           train_dataloader, metrics,lr_scheduler)

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

    return {"train_loss": train_loss_list, "val_loss": val_loss_list}


class Job:
    def __init__(self):
        self.log_file = utils.set_logger("./train.log")
        self.device = utils.get_device()
        self.batch_size = 32
        self.epoches = 50
        self.lr = 0.000001
        self.bert_model = model
        self.bert_tokenizer = tokenizer
        self.text_json = "training.en-en.data"
        self.label_json = "training.en-en.gold"
        self.num_class = 2
        self.window_size = 5
        self.dropout = 0.2
        self.in_features = 768
        self.loss_result = None
        if self.window_size < 0:
            if self.dropout > 0:
                self.model_dir = "End2endBertNet_nochop_withdropout"
            else:
                self.model_dir = "End2endBertNet_nochop_nodropout"
        else:
            if self.dropout > 0:
                self.model_dir = "End2endBertNet_chop5_withdropout"
            else:
                self.model_dir = "End2endBertNet_chop5_nodropout"

    def train(self):
        self.trainX, self.trainY, self.testX, self.testY = split_dataSet(self.text_json, self.label_json, self.window_size)
        train_data = MyData.from_list(self.trainX,self.trainY)
        valid_data = MyData.from_list(self.testX,self.testY)

        train_dataloader = DataLoader(dataset=train_data, sampler=RandomSampler(
            train_data), batch_size=self.batch_size, shuffle=False, num_workers=0, collate_fn=collect_func, drop_last=False)

        valid_dataloader = DataLoader(
            dataset=valid_data, batch_size=self.batch_size//2, shuffle=False, num_workers=0,collate_fn=collect_func, drop_last=False)
        model = WordDisambiguationNet(bert_model=self.bert_model,bert_tokenizer=self.bert_tokenizer,in_features=self.in_features,dropout=self.dropout)
        model.to(device=self.device)
        # print(model)
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

        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps= 200, num_training_steps=self.epoches*len(train_dataloader))

        criterion = nn.CrossEntropyLoss()
        self.loss_result = train_and_evaluate(model, train_dataloader, valid_dataloader, optimizer,
                                              criterion, utils.metrics, self.epoches, self.model_dir, lr_scheduler, restore_file=None)
        curr_hyp = {"epochs": self.epoches, "batch_size": self.batch_size,"lr": self.lr}
        utils.save_dict_to_json(curr_hyp, os.path.join(
            self.model_dir, "train_hyp.json"))
        df = pd.DataFrame(
            data={'val': self.loss_result['val_loss'], 'train': self.loss_result['train_loss']})
        df.to_csv("{}/loss.csv".format(self.model_dir))

    def predict(self):

        valid_data = MyData(self.testX, self.testY)
        valid_dataloader = DataLoader(dataset=valid_data, batch_size=len(
            valid_data), shuffle=False, num_workers=0, drop_last=False)
        model = self.model
        utils.load_checkpoint(os.path.join(
            self.model_dir, "best.pth.tar"), model)
        model.to(device=self.device)
        model.eval()
        print("-----------------开始对{}模型在测试集上进行预测-----------------".format(self.model_dir))
        for batch in valid_dataloader:
            inputX, inputY = batch
            inputX = inputX.to(self.device)
            inputY = inputY.to(self.device)
            y_pred = np.argmax(
                model(inputX).data.cpu().numpy(), axis=1).squeeze()
            y_true = inputY.data.cpu().numpy().squeeze()
            result = pd.DataFrame(
                data={'y_true': y_true, 'y_pred': y_pred}, index=range(len(y_pred)))
            result.to_csv(r"{}/result.csv".format(self.model_dir))
            # print("r2:{},rmse:{}"(y_pred,y_true),utils.rmse(y_pred,y_true)))
            print("--测试集上的性能指标f1:{},acc:{}".format(f1_score(y_true,
                                                            y_pred), accuracy_score(y_true, y_pred)))

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

    def plot_loss(self):
        fig = plt.figure(figsize=(10,8))
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
        plt.show()
        fig.savefig('loss_plot.png', bbox_inches='tight')


if __name__ == "__main__":
    job = Job()
    job.train()
    # sentence = "The WTO, uniting some 140 States, must now be thoroughly reformed in order to really liberalize and democratize trade relations."
    # lemma = "unite"
    # start,end = 9,16
    # print(process_sentence(sentence,lemma,-5,start,end))