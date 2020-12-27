import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.nn import init

def layer_init(layer):
    if isinstance(layer,nn.Conv2d):
        init.xavier_uniform_(layer.weight.data)
        init.constant_(layer.bias.data,0.1)
    if isinstance(layer,nn.BatchNorm2d):
        layer.weight.data.fill_(1)
        layer.bias.data.zero_()
    if isinstance(layer,nn.Linear):
        init.xavier_normal_(layer.weight.data)
        init.constant_(layer.bias.data,0.0)
    if isinstance(layer,nn.Sequential):
        for i in range(len(layer)):
            layer_init(layer[i])
    return True

class TextCNN(nn.Module):
    def __init__(self,params,word_embedding=None):
        super(TextCNN,self).__init__()
        self.n_class = params.n_class
        self.embed_dim = params.embed_dim
        self.vocab_size = params.vocab_size
        self.seq_max_length = params.seq_max_length
        self.out_channels = 64
        self.hidden_size = 100

        # embedding_static = nn.Embedding(vocab_size,embed_dim)
        embedding_non_static = nn.Embedding(self.vocab_size,self.embed_dim)
        self.embedding_non_static = embedding_non_static.from_pretrained(word_embedding,freeze=True)

        self.ngram3_layer = nn.Sequential(nn.Conv1d(self.embed_dim,self.out_channels,3),nn.BatchNorm1d(self.out_channels),nn.ReLU(inplace=True),nn.MaxPool1d(self.seq_max_length-2))
        self.ngram4_layer = nn.Sequential(nn.Conv1d(self.embed_dim,self.out_channels,4),nn.BatchNorm1d(self.out_channels),nn.ReLU(inplace=True),nn.MaxPool1d(self.seq_max_length-3))
        self.ngram5_layer = nn.Sequential(nn.Conv1d(self.embed_dim,self.out_channels,5),nn.BatchNorm1d(self.out_channels),nn.ReLU(inplace=True),nn.MaxPool1d(self.seq_max_length-4))
        self.fc_layer = nn.Sequential(nn.Linear(self.out_channels*3,self.hidden_size),nn.ReLU(inplace=True),nn.Dropout(0.5),nn.Linear(self.hidden_size,self.n_class))
        # self.fc_layer.apply(layer_init)

    def forward(self,inputX):
        embed = self.embedding_non_static(inputX)
        embed = embed.permute(0,2,1)
        out2 = (self.ngram3_layer(embed)).squeeze(2)
        out3 = (self.ngram4_layer(embed)).squeeze(2)
        out4 = (self.ngram5_layer(embed)).squeeze(2)
        out = self.fc_layer(torch.cat((out2,out3,out4),dim=1))
        return out

class BiRNN(nn.Module):
    def __init__(self,params,word_embedding):
        super(BiRNN,self).__init__()
        self.vocab_size = params.vocab_size
        self.embed_dim = params.embed_dim
        self.n_class = params.n_class
        self.seq_max_length = params.seq_max_length
        self.hidden_size = 100

        self.embedding = nn.Embedding(self.vocab_size,self.embed_dim)
        self.embedding = self.embedding.from_pretrained(word_embedding)

        self.encoder = nn.GRU(input_size=self.embed_dim,hidden_size=self.hidden_size,bidirectional=True,batch_first=True,num_layers=6)

        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),nn.ReLU(inplace=True),nn.Linear(self.hidden_size,self.n_class))

    def forward(self, inputX):
        embed = self.embedding(inputX)
        out,hn = self.encoder(embed)
        # out = out[:,-1,:].squeeze(2)
        return self.fc_layer(out[:,-1,:])


class Fasttext(nn.Module):
    def __init__(self,params,embedding_matrix=None):
        super(Fasttext,self).__init__()
        self.vocab_size,self.embed_dim = embedding_matrix.shape
        self.n_class = params.n_class
        self.hidden_size = 100
        self.seq_max_length = params.seq_max_length

        self.embedding = nn.EmbeddingBag(self.vocab_size,self.embed_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.fc_layer = nn.Sequential(nn.Linear(self.embed_dim,self.hidden_size),nn.ReLU(inplace=True),nn.Linear(self.hidden_size,self.n_class))

    def forward(self, inputX):  
        embed = self.embedding(inputX)
        return self.fc_layer(embed)


class RCNN(nn.Module):
    def __init__(self,params,embedding_matrix=None):
        super(RCNN,self).__init__()
        self.vocab_size,self.embed_dim = params.vocab_size,params.embed_dim
        self.n_class = params.n_class
        self.hidden_size = 200
        self.seq_max_length = params.seq_max_length

        embedding = nn.Embedding(self.vocab_size,self.embed_dim)
        self.embedding = embedding.from_pretrained(embedding_matrix)

        self.rnn = nn.GRU(input_size=self.embed_dim,hidden_size=self.hidden_size,num_layers=4,batch_first=True,bidirectional=True)
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=2*self.hidden_size+self.embed_dim,out_channels=64,kernel_size=1),nn.BatchNorm1d(64),nn.MaxPool1d(self.seq_max_length))
        self.fc_layer = nn.Sequential(nn.Linear(64,self.n_class),nn.ReLU(inplace=True),nn.Dropout(0.5))

    def forward(self, inputX):
        embed = self.embedding(inputX)
        out,hn = self.rnn(embed)
        
        input1 = torch.cat((out[:,:,:self.hidden_size],embed,out[:,:,self.hidden_size:]),dim=2)
        # print("input1's shape:{}".format(input1.shape))
        input1 = input1.permute(0,2,1)
        # print("input1's shape:{}".format(input1.shape))

        out1 = self.cnn(input1).squeeze(2)
        # print("out1's shape:{}".format(out1.shape))
        return self.fc_layer(out1)

class AttentionRNN(nn.Module):
    def __init__(self,params,embedding_matrix):
        super(AttentionRNN,self).__init__()
        self.vocab_size = params.vocab_size
        self.embed_dim = params.embed_dim
        self.n_class = params.n_class
        # self.embedding_matrix = embedding_matrix
        self.seq_max_length = params.seq_max_length
        self.hidden_size = 100

        self.att_w = nn.Parameter(torch.Tensor(1,1,self.hidden_size))
        nn.init.normal_(self.att_w)

        embedding = nn.Embedding(self.vocab_size,self.embed_dim)
        self.embdedding = embedding.from_pretrained(embedding_matrix)
        self.encoder = nn.GRU(input_size=self.embed_dim,hidden_size=self.hidden_size,batch_first=True,bidirectional=True)
        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size,self.n_class),nn.ReLU(inplace=True))
        # self.att_layer = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputX):
        embed = self.embdedding(inputX)
        out,hn = self.encoder(embed)
        # out :B*S*2H
        # hn :B*(num_layers*bidirectional)*H
        out_left,out_right = out[:,:,0:self.hidden_size],out[:,:,self.hidden_size:]
        out_add = out_left + out_right #将前向与后向相加 B*T*E
        M = nn.Tanh()(out_add) # B*T*E
        B = M.shape[0] # 计算batch_size
        # att_W = nn.Parameter(torch.Tensor(B,1,self.embed_dim)) # B*1*E
        att_W = self.att_w.expand(B,1,self.hidden_size)
        # print("att_w's device:{}".format(att_W.device))
        # print("M's shape:{},att_W's shape:{}".format(M.shape,att_W.shape))
        alpha = self.softmax(torch.bmm(M,att_W.permute(0,2,1))) # (B*T*E)*(B*E*1)=(B*T*1)
        context = torch.bmm(alpha.permute(0,2,1),out_add) # (B*1*T) * (B*T*E) = (B*1*E)
        return self.fc_layer(context.squeeze(1))


class HAN(nn.Module):
    def __init__(self,vocab_size,embed_dim,seq_max_length,n_class,embedding_matrix):
        super(HAN,self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_max_length = seq_max_length
        self.n_class = n_class
        self.hidden_size = 100
        self.num_layers = 4
    
        self.att_w = nn.Parameter(torch.Tensor(self.hidden_size,1)) # 
        nn.init.normal_(self.att_w)

        self.embedding = nn.Embedding(self.vocab_size,self.embed_dim)
        self.att_layer = nn.Sequential(nn.Linear(self.hidden_size,self.hidden_size),nn.Tanh())
        self.gru = nn.GRU(input_size=self.embed_dim,hidden_size=self.hidden_size,num_layers=self.num_layers,batch_first=True,bidirectional=True)

        self.fc_layer = nn.Linear(self.hidden_size,self.n_class)

    def calc_attention(self,hidden_state,seq_length):
        def _sim(u,v):
            ss = 0.0
            for i in range(len(u)):
                ss += u[i]*v[i]
            return ss
        u_list,alpha_list = [],[]
        for i in range(seq_length):
            u_i = hidden_state[:,i,:]
            att = torch.dot(self.att_layer(u_i),self.att_w)
            alpha_list.append(att)
        for i in range(seq_length):
            u_i = u_list[i]
            for j in range(seq_length):
                u_j = u_list[j]
            alpha_list.append(_sim(u_i,u_j))
        alpha_tensor = F.softmax(torch.tensor(alpha_list))
        result = None
        for i in range(seq_length):
            result += alpha_tensor[i]* hidden_state[:,i,:]
        return result

    def forward(self,inputX):
        embed = self.embdedding(inputX)
        out,hn = self.gru(embed)
        out_left,out_right = out[:,:,0:self.hidden_size],out[:,:,self.hidden_size:]
        out_add = out_left + out_right #将前向与后向相加 B*S*H
        u_list = []
        for i in range(self.seq_max_length):
            u_i = out_add[:,i,:]
            att = self.att_layer(u_i)
            u_list.append(att.tolist())
        
class SimNet(nn.Module):
    def __init__(self,vocab_size,embed_dim,n_class,embedding_matrix):
        super(SimNet,self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_class = n_class
        self.embedding_matrix = embedding_matrix

        self.encoder = nn.Sequential(nn.Embedding(self.vocab_size,self.embed_dim),nn.GRU(input_size=self.embed_dim,hidden_size=128,num_layers=4,batch_first=True,bidirectional=True))

        self.sim_loss = nn.CosineSimilarity(dim=1)

    def forward(self,texta,textb):
        out1,hn1 = self.encoder(texta)
        out2,hn2 = self.encoder(textb)
        h1 = out1[:,-1,:]
        h2 = out2[:,-1,:]
        out = self.sim_loss(h1,h2)
        return out

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'): 
        super(LabelSmoothingCrossEntropy,self).__init__()
        self.eps,self.reduction = eps,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum': 
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':  
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class Transformer(nn.Module):
    def __init__(self,vocab_size,num_layers,n_class,d_model=512,n_head=8):
        super(Transformer,self).__init__()
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.fc_layer = nn.Sequential(
            nn.Linear(d_model,n_class),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        embed = self.embdedding(x)
        out1 = self.encoder(embed)
        return self.fc_layer(out1)

