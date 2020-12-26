import torch
import torch.nn as nn
from utils import get_device

class BertClassfier(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, bert_config, num_class, dropout=0.2):
        super(BertClassfier, self).__init__()
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.dropout = nn.Dropout(dropout)
        self.num_class = num_class
        self.fc1 = nn.Linear(
            bert_config.hidden_size, bert_config.hidden_size)
        self.fc2 = nn.Linear(bert_config.hidden_size, self.num_class)

    def forward(self, inputs):
        encoder_inputs = self.bert_tokenizer(inputs,return_tensors='pt',padding=True)
        encoder_inputs = encoder_inputs.to(get_device())
        # 句向量 [batch_size,hidden_size]
        bert_out = self.bert_model(**encoder_inputs)[1]
        bert_out = self.dropout(bert_out)
        bert_out = self.fc1(bert_out)
        bert_out = self.dropout(bert_out)
        bert_out = self.fc2(bert_out)  # [batch_size,num_class]
        return bert_out

class BertSequenceClassfier(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, bert_config, num_class, pool_type='avg', dropout=0.2):
        super(BertSequenceClassfier, self).__init__()
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.dropout = nn.Dropout(dropout)
        self.num_class = num_class
        self.pool_type = pool_type
        self.fc1 = nn.Linear(
            bert_config.hidden_size, bert_config.hidden_size)
        self.fc2 = nn.Linear(bert_config.hidden_size, self.num_class)
        if self.pool_type == 'avg':
            self.pool_layer = nn.AdaptiveAvgPool1d(1)
        if self.pool_type == 'max':
            self.pool_type = nn.AdaptiveMaxPool1d(1)

    def forward(self, inputs):
        encoder_inputs = self.bert_tokenizer(inputs,return_tensors='pt',padding=True)
        encoder_inputs = encoder_inputs.to(get_device())
        # 句向量 [batch_size,hidden_size]
        bert_out = self.bert_model(**encoder_inputs)[0]
        if self.pool_type == 'CLS':
            bert_out = bert_out[:,0,:]
        else:
            bert_out = bert_out.permute(0,2,1)[:,1:-1,:]
            bert_out = self.pool_layer(bert_out).squeeze(2)
        bert_out = self.dropout(bert_out)
        bert_out = self.fc1(bert_out)
        bert_out = self.dropout(bert_out)
        bert_out = self.fc2(bert_out)  # [batch_size,num_class]
        return bert_out
