import torch
import torch.nn as nn

class Attention(nn.Module):
    """
    参考Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    具体计算公式如下:
    input:        H = (h1,h2,...,hn) n为序列长度
    M:            M = tanh(H)
    W:            ei = softmax(W^T*M) i = 1,2,3,...,n
    output:       out = sum(ei*hi)  E^T*H ->(B,D)
    """
    def __init__(self, feature_dim, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.weight = nn.Parameter(torch.zeros(feature_dim,1))
        nn.init.xavier_normal_(self.weight)

    def forward(self,hidden):
        M = torch.tanh(hidden) # B*L*D
        U = torch.matmul(M,self.W).squeeze(2)
        E = torch.softmax(U,dim=1) # B*L
        # E(B,1,L)*(B,L,D) = (B,1,D)->(B,D)
        O = torch.bmm(E.unsqueeze(1),M).squeeze(1)
        return O 


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock,self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0,1),value=0),
            nn.MaxPool1d(kernel_size=3,stride=2)
            # nn.AvgPool1d(kernel_size=3,stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channel_size,self.channel_size,kernel_size=3,padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channel_size,self.channel_size,kernel_size=3,padding=1)
        )
    def forward(self,x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        return x + x_shortcut


class DPCNN(nn.Module):
    def __init__(self,filter_num,seq_length,embed_dim,num_class):
        super(DPCNN,self).__init__()
        self.out_channels = filter_num
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.region_layer = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim,out_channels=self.out_channels,kernel_size=3,padding=1),
            nn.BatchNorm1d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)   
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.out_channels,out_channels=self.out_channels,kernel_size=3,padding=1),
            nn.BatchNorm1d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.out_channels,out_channels=self.out_channels,kernel_size=3,padding=1)
        )
        resnet_block_list = []
        while self.seq_length > 3:
            resnet_block_list.append(ResnetBlock(self.out_channels))
            self.seq_length = self.seq_length//2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.MaxPool = nn.AdaptiveMaxPool1d(1)
        # self.MaxPool = nn.AdaptiveAvgPool1d(1)

    def forward(self,inputX):
        # inputX = inputX.unsqueeze(2) #dim = 1
        assert inputX.shape[2] == self.embed_dim
        inputX = inputX.permute(0,2,1)
        out = self.region_layer(inputX)
        out = self.conv_block(out)
        out = self.resnet_layer(out)
        out = self.MaxPool(out)
        # print("out's shape:{}".format(out.shape))
        return out




class TextCNN(nn.Module):
    def __init__(self, kernel_sizes, num_channels, embed_size, num_class, dropout=0.2):
        super(TextCNN,self).__init__()
        self.embed_size = embed_size
        self.kernel_sizes = kernel_sizes
        self.num_channels = num_channels
        self.num_class = num_class
        self.encoder_layers = nn.ModuleList()
        for kernel,channel in zip(self.kernel_sizes,self.num_channels):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=self.embed_size,out_channels=channel,kernel_size=kernel),
                    nn.BatchNorm1d(num_features=channel),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveMaxPool1d(1)
                )
            )
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=sum(self.num_channels),out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256,out_features=num_class),
            nn.Dropout(p=dropout)
        )
    
    def forward(self,inputs):
        """
        inputs 的形状为(B,seq_length,embed_size)
        """
        assert inputs.shape[2] == self.embed_size
        inputs = inputs.permute(0,2,1)
        out = torch.cat([layer(inputs).squeeze(2) for layer in self.encoder_layers],dim=1)
        return self.fc_layer(out)


class BiGRU(nn.Module):
    def __init__(self,hidden_size,num_class,embed_size,num_layers):
        super(BiGRU,self).__init__()
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers

        self.encoder = nn.GRU(input_size=self.embed_size,hidden_size=self.hidden_size,bidirectional=True,batch_first=True,num_layers=self.num_layers)

        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),nn.ReLU(inplace=True),nn.Linear(self.hidden_size,self.num_class))

    def forward(self,inputs):
        assert inputs.shape[2] == self.embed_size
        out,hn = self.encoder(inputs)
        concat = torch.cat((hn[0],hn[1]),dim=1)
        return self.fc_layer(concat)

class AttentionBiGRU(nn.Module):
    def __init__(self,hidden_size,num_class,embed_size,num_layers):
        super(AttentionBiGRU,self).__init__()
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers

        self.encoder = nn.GRU(input_size=self.embed_size,hidden_size=self.hidden_size,bidirectional=True,batch_first=True,num_layers=self.num_layers)

        self.att_layer = Attention(2*self.hidden_size)

        self.fc_layer = nn.Sequential(nn.Linear(self.hidden_size*2,self.hidden_size),nn.ReLU(inplace=True),nn.Linear(self.hidden_size,self.num_class))

    def forward(self,inputs):
        assert inputs.shape[2] == self.embed_size
        out,hn = self.encoder(inputs)
        att = self.att_layer(out)
        return self.fc_layer(att)