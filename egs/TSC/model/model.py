import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,n_head=1,num_layers=1):
        super(MultiHeadSelfAttention,self).__init__()
        self.num_layers = num_layers
        self.n_head = n_head
        self.d_model = d_model
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,nhead=self.n_head)
        self.attention_layer = nn.TransformerEncoder(self.encoder_layer,num_layers=self.num_layers)
        self.W = nn.Parameter(torch.Tensor(7,1))
        nn.init.xavier_normal_(self.W) #初始化,必须添加

    def forward(self,inputs):
        """
        (B,L,H) -> (B,L,H) -> (B,H,L) -> (B,H,1) -> (B,H)
        """
        out = self.attention_layer(inputs)
        out = out.permute(0,2,1)
        out = out.matmul(self.W)
        return out.squeeze(2)

class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock,self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0,1),value=0),
            # nn.MaxPool1d(kernel_size=3,stride=2)
            nn.AvgPool1d(kernel_size=3,stride=2)
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


class TimeSeriesFeature(nn.Module):
    def __init__(self,filter_num,seq_length,num_class):
        super(TimeSeriesFeature,self).__init__()
        self.out_channels = filter_num
        self.seq_length = seq_length
        self.region_layer = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=self.out_channels,kernel_size=3,padding=1),
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
        # self.MaxPool = nn.AdaptiveMaxPool1d(1)
        self.MaxPool = nn.AdaptiveAvgPool1d(1)

    def forward(self,inputX):
        inputX = inputX.unsqueeze(2) #dim = 1
        inputX = inputX.permute(0,2,1)
        out = self.region_layer(inputX)
        out = self.conv_block(out)
        out = self.resnet_layer(out)
        out = self.MaxPool(out)
        # print("out's shape:{}".format(out.shape))
        return out

class SequenceClassify(nn.Module):
    def __init__(self,out_channels,num_class,hidden_num,seq_lengths,mode='att'):
        super(SequenceClassify,self).__init__()
        assert len(seq_lengths) == 7
        self.out_channels = out_channels
        self.num_class = num_class
        self.hidden_num = hidden_num
        self.seq_lengths = seq_lengths
        self.mode = mode
        self.feature_layer1 = TimeSeriesFeature(filter_num=self.out_channels,seq_length=self.seq_lengths[0],num_class=self.num_class)
        self.feature_layer2 = TimeSeriesFeature(filter_num=self.out_channels,seq_length=self.seq_lengths[1],num_class=self.num_class)
        self.feature_layer3 = TimeSeriesFeature(filter_num=self.out_channels,seq_length=self.seq_lengths[2],num_class=self.num_class)
        self.feature_layer4 = TimeSeriesFeature(filter_num=self.out_channels,seq_length=self.seq_lengths[3],num_class=self.num_class)
        self.feature_layer5 = TimeSeriesFeature(filter_num=self.out_channels,seq_length=self.seq_lengths[4],num_class=self.num_class)
        self.feature_layer6 = TimeSeriesFeature(filter_num=self.out_channels,seq_length=self.seq_lengths[5],num_class=self.num_class)
        self.feature_layer7 = TimeSeriesFeature(filter_num=self.out_channels,seq_length=self.seq_lengths[6],num_class=self.num_class)

        self.fc_layer = nn.Sequential(
            nn.Linear(self.out_channels,self.hidden_num),
            nn.BatchNorm1d(self.hidden_num),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_num,self.num_class),
            nn.Dropout(0.2)
        )
        self.W = nn.Parameter(torch.Tensor(7,1))
        nn.init.xavier_normal_(self.W) #初始化,必须添加
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.att_layer = MultiHeadSelfAttention(d_model=self.out_channels,n_head=1,num_layers=1)

        
    def forward(self,x1,x2,x3,x4,x5,x6,x7):
        f1 = self.feature_layer1(x1)
        f2 = self.feature_layer2(x2)
        f3 = self.feature_layer3(x3)
        f4 = self.feature_layer4(x4)
        f5 = self.feature_layer5(x5)
        f6 = self.feature_layer6(x6)
        f7 = self.feature_layer7(x7)
        # 7*(B,H) -> (B,H,7)
        concat = torch.cat((f1,f2,f3,f4,f5,f6,f7),dim=2)
        if self.mode == 'avg':
            out = self.avgpool(concat)
            return self.fc_layer(out.squeeze(2))
        elif self.mode == 'att':
            concat = concat.permute(0,2,1)
            out = self.att_layer(concat)
            return self.fc_layer(out)
        else:
            out = concat.matmul(self.W)
            return self.fc_layer(out.squeeze(2))
        


