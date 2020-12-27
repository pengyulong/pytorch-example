import torch
import torch.nn as nn
from utils import get_device
from .layer import TextCNN,BiGru

class BertClassfier(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, bert_config, num_class, dropout=0.2):
        super(BertClassfier, self).__init__()
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.num_class = num_class
        self.fc_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=bert_config.hidden_size,out_features=bert_config.hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(bert_config.hidden_size,self.num_class)
        )

    def forward(self, inputs):
        encoder_inputs = self.bert_tokenizer(inputs,return_tensors='pt',padding=True)
        encoder_inputs = encoder_inputs.to(get_device())
        # 句向量 [batch_size,hidden_size]
        bert_out = self.bert_model(**encoder_inputs)[1]
        bert_cout = self.fc_layer(bert_cout)
        return bert_out

class BertSequenceClassfier(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, bert_config, num_class, pool_type='avg', dropout=0.2):
        super(BertSequenceClassfier, self).__init__()
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.num_class = num_class
        self.pool_type = pool_type
        if self.pool_type == 'avg':
            self.pool_layer = nn.AdaptiveAvgPool1d(1)
        if self.pool_type == 'max':
            self.pool_type = nn.AdaptiveMaxPool1d(1)
        self.fc_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=bert_config.hidden_size,out_features=bert_config.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(bert_config.hidden_size,self.num_class)
        )
    def forward(self, inputs):
        encoder_inputs = self.bert_tokenizer(inputs,return_tensors='pt',padding=True)
        encoder_inputs = encoder_inputs.to(get_device())
        # 句向量 [batch_size,hidden_size]
        bert_out = self.bert_model(**encoder_inputs)[0]
        if self.pool_type == 'CLS':
            bert_out = bert_out[:,0,:]
        else:
            bert_out = bert_out[:,1:-1,:].permute(0,2,1)
            bert_out = self.pool_layer(bert_out).squeeze(2)
        bert_out = self.fc_layer(bert_out)
        return bert_out

class BertEncoderClassfier(nn.Module):
    def __init__(self, bert_model, bert_tokenizer, num_class, bert_config, encoder_type='cnn', dropout=0.2):
        super(BertEncoderClassfier, self).__init__()
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.dropout = nn.Dropout(dropout)
        self.num_class = num_class
        if encoder_type == 'cnn':
            self.encoder_layer = TextCNN(kernel_sizes=range(2,5),num_channels=[100,100,100,100],embed_size=bert_config.hidden_size,num_class=self.num_class,dropout=dropout)
        if encoder_type == 'rnn':
            self.encoder_layer = BiGru(hidden_size=128,num_class=num_class,embed_size=bert_config.hidden_size,num_layers=2)

    def forward(self, inputs):
        encoder_inputs = self.bert_tokenizer(inputs,return_tensors='pt',padding=True)
        encoder_inputs = encoder_inputs.to(get_device())
        # 句向量 [batch_size,hidden_size]
        bert_out = self.bert_model(**encoder_inputs)[0]
        bert_out = bert_out[:,1:-1,:].permute(0,2,1)
        bert_out = self.encoder_layer(bert_out)
        return bert_out

