import torch
import torch.nn as nn
from utils import get_device
from .layer import DPCNN,AttentionBiGRU,BiGRU

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
            self.pool_layer = nn.AdaptiveMaxPool1d(1)
        self.fc_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=bert_config.hidden_size,out_features=bert_config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bert_config.hidden_size,self.num_class)
        )
    def forward(self, inputs):
        encoder_inputs = self.bert_tokenizer(inputs,return_tensors='pt',padding=True)
        encoder_inputs = encoder_inputs.to(get_device())
        if self.pool_type in ['avg','max','CLS']:
            bert_out = self.bert_model(**encoder_inputs)[0]
            if self.pool_type == 'CLS':
                out = bert_out[:,0,:]
            else:
                out = self.pool_layer(bert_out[:,1:-1,:].permute(0,2,1)).squeeze(2)
        else:
            out = self.bert_model(**encoder_inputs)[1]
        # out 为句向量:[batch_size,hidden_size]
        return self.fc_layer(out)        

class BertEncoderClassfier(nn.Module):
    """BiGRU, Att_BiGRU, DPCNN
    """
    def __init__(self, bert_model, bert_tokenizer, num_class, bert_config, max_length = 510, encoder_type=None, dropout=0.2):
        super(BertEncoderClassfier, self).__init__()
        self.device = get_device()
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_config = bert_config
        self.dropout = nn.Dropout(dropout)
        self.num_class = num_class
        self.max_length = max_length
        if encoder_type == 'DPCNN':
            self.encoder_layer = DPCNN(filter_num=128,seq_length=max_length,embed_dim=self.bert_config.hidden_size,num_class=self.num_class).to(self.device)
            # self.encoder_layer = TextCNN(kernel_sizes=range(2,5),num_channels=[128,128,128],embed_size=self.bert_config.hidden_size,num_class=self.num_class,dropout=dropout).to(self.device)
        if encoder_type == 'BiGRU':
            self.encoder_layer = BiGRU(hidden_size=128,num_class=num_class,embed_size=bert_config.hidden_size,num_layers=1).to(self.device)
        if encoder_type == "Att_BiGRU":
            self.encoder_layer = AttentionBiGRU(hidden_size=128,num_class=self.num_class,embed_size=bert_config.hidden_size,num_layers=1).to(self.device)

        self.fc_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=128,out_features=bert_config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bert_config.hidden_size,self.num_class)
        )

    def forward(self, inputs):
        encoder_inputs = self.bert_tokenizer(inputs,return_tensors='pt',padding='max_length',truncation=True,max_length=self.max_length)
        encoder_inputs = encoder_inputs.to(get_device())
        # 句向量 [batch_size,hidden_size]
        bert_out = self.bert_model(**encoder_inputs)[0]
        bert_out = bert_out[:,1:-1,:]
        bert_out = self.encoder_layer(bert_out)
        return bert_out

