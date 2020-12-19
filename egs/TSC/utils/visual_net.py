import torch.functional as F
from sklearn import preprocessing
import torch.nn as nn
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler)
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data.dataloader import default_collate
import utils
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from torch.optim import lr_scheduler
import pandas as pd
from sklearn.manifold import TSNE


def load_dataSet(inputX, target):
    trainX, testX, trainY, testY = train_test_split(
        inputX, target, test_size=0.1, random_state=0)
    return trainX, trainY, testX, testY


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head=1, num_layers=1):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_layers = num_layers
        self.n_head = n_head
        self.d_model = d_model
        self.seq_length = seq_length
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_head)
        self.attention_layer = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.num_layers)
        # self.avgpool = nn.AvgPool1d(self.seq_length)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputs):
        """
        (B,L,D)->(B,1,D)
        """
        out = self.attention_layer(inputs)
        out = out.permute(0, 2, 1)
        out = self.avgpool(out)
        return out.squeeze(2)


class ResnetBlock(nn.Module):
    def __init__(self, channel_size):
        super(ResnetBlock, self).__init__()
        self.channel_size = channel_size
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            # nn.MaxPool1d(kernel_size=3,stride=2)
            nn.AvgPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.channel_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.channel_size, self.channel_size,
                      kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)
        x = self.conv(x_shortcut)
        return x + x_shortcut


def get_data(csvfile):
    dataSet = pd.read_csv(csvfile, index_col=0, header=None)
    dataX = np.array(dataSet.iloc[:, :-1])
    seq_length = dataX.shape[1]
    dataX = normalize(dataX, axis=1, norm='max')
    # normal = dataX.min(axis=1).reshape((-1,1))
    # dataX = dataX - normal

    dataY = np.array(dataSet.iloc[:, -1]-1)
    # dataY = dataY.reshape(-1,1)
    trainX, trainY, testX, testY = load_dataSet(dataX, dataY)
    return trainX, trainY, testX, testY, seq_length


class TimeSeriesFeature(nn.Module):
    def __init__(self, filter_num, seq_length, num_class):
        super(TimeSeriesFeature, self).__init__()
        self.out_channels = filter_num
        self.seq_length = seq_length
        self.region_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.out_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.out_channels,
                      out_channels=self.out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.out_channels,
                      out_channels=self.out_channels, kernel_size=3, padding=1)
        )
        resnet_block_list = []
        while self.seq_length > 3:
            resnet_block_list.append(ResnetBlock(self.out_channels))
            self.seq_length = self.seq_length//2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        # self.MaxPool = nn.AdaptiveMaxPool1d(1)
        self.MaxPool = nn.AdaptiveAvgPool1d(1)

    def forward(self, inputX):
        inputX = inputX.unsqueeze(2)  # dim = 1
        inputX = inputX.permute(0, 2, 1)
        out = self.region_layer(inputX)
        out = self.conv_block(out)
        out = self.resnet_layer(out)
        out = self.MaxPool(out)
        # print("out's shape:{}".format(out.shape))
        return out


class SequenceData(Dataset):
    def __init__(self, dataX1, dataX2, dataX3, dataX4, dataX5, dataX6, dataX7, dataY):
        self.dataX1 = torch.Tensor(dataX1)
        self.dataX2 = torch.Tensor(dataX2)
        self.dataX3 = torch.Tensor(dataX3)
        self.dataX4 = torch.Tensor(dataX4)
        self.dataX5 = torch.Tensor(dataX5)
        self.dataX6 = torch.Tensor(dataX6)
        self.dataX7 = torch.Tensor(dataX7)
        # one_hot = OneHotEncoder(sparse=False)
        # dataY = one_hot.fit_transform(dataY)
        # self.dataY = torch.Tensor(dataY)
        self.dataY = torch.LongTensor(dataY)

    def __len__(self):
        return len(self.dataY)

    def __getitem__(self, index):
        return self.dataX1[index], self.dataX2[index], self.dataX3[index], self.dataX4[index], self.dataX5[index], self.dataX6[index], self.dataX7[index], self.dataY[index]


class SequenceClassify(nn.Module):
    def __init__(self, out_channels, num_class, hidden_num, seq_lengths):
        super(SequenceClassify, self).__init__()
        assert len(seq_lengths) == 7
        self.out_channels = out_channels
        self.num_class = num_class
        self.hidden_num = hidden_num
        self.seq_lengths = seq_lengths
        self.feature_layer1 = TimeSeriesFeature(
            filter_num=self.out_channels, seq_length=self.seq_lengths[0], num_class=self.num_class)
        self.feature_layer2 = TimeSeriesFeature(
            filter_num=self.out_channels, seq_length=self.seq_lengths[1], num_class=self.num_class)
        self.feature_layer3 = TimeSeriesFeature(
            filter_num=self.out_channels, seq_length=self.seq_lengths[2], num_class=self.num_class)
        self.feature_layer4 = TimeSeriesFeature(
            filter_num=self.out_channels, seq_length=self.seq_lengths[3], num_class=self.num_class)
        self.feature_layer5 = TimeSeriesFeature(
            filter_num=self.out_channels, seq_length=self.seq_lengths[4], num_class=self.num_class)
        self.feature_layer6 = TimeSeriesFeature(
            filter_num=self.out_channels, seq_length=self.seq_lengths[5], num_class=self.num_class)
        self.feature_layer7 = TimeSeriesFeature(
            filter_num=self.out_channels, seq_length=self.seq_lengths[6], num_class=self.num_class)

        self.fc_layer = nn.Sequential(
            nn.Linear(self.out_channels, self.hidden_num),
            nn.BatchNorm1d(self.hidden_num),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_num, self.num_class),
            nn.Dropout(0.2)
        )
        self.W = nn.Parameter(torch.Tensor(7, 1))
        nn.init.xavier_normal_(self.W)  # 初始化,必须添加
        # self.att_layer = SelfAttention(hidden_dim=self.out_channels)
        # self.att_layer = MultiHeadSelfAttention(d_model=out_channels)

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        # batch_size,Seq_len = inputX.shape
        f1 = self.feature_layer1(x1)
        f2 = self.feature_layer2(x2)
        f3 = self.feature_layer3(x3)
        f4 = self.feature_layer4(x4)
        f5 = self.feature_layer5(x5)
        f6 = self.feature_layer6(x6)
        f7 = self.feature_layer7(x7)
        concat = torch.cat((f1, f2, f3, f4, f5, f6, f7), dim=2)
        # print("concat' s shape:{}".format(concat.shape))
        # concat = concat.permute(0,2,1)

        # out = self.att_layer(concat)

        out = concat.matmul(self.W)
        out = out.squeeze(2)

        # out = self.att_layer(concat)
        return self.fc_layer(out)


def show_feature(feature, labels, fig_name):
    x_embed = TSNE(n_components=3).fit_transform(feature)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x_embed_0 = x_embed[labels == 0, :]  # 类别0
    x_embed_1 = x_embed[labels == 1, :]  # 类别1
    x_embed_2 = x_embed[labels == 2, :]  # 类别3
    p0 = ax.scatter(x_embed_0[:, 0], x_embed_0[:, 1], x_embed_0[:, 2],
                    color='r', cmap='brg', marker='o', depthshade=False, label='class 0')
    p1 = ax.scatter(x_embed_1[:, 0], x_embed_1[:, 1], x_embed_1[:, 2],
                    color='g', cmap='brg', marker='o', depthshade=False, label='class 1')
    p2 = ax.scatter(x_embed_2[:, 0], x_embed_2[:, 1], x_embed_2[:, 2],
                    color='b', cmap='brg', marker='o', depthshade=False, label='class 2')
    ax.legend()
    # ax.set_title(title_str)
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    # plt.show()


if __name__ == "__main__":

    trainX1, trainY, testX1, testY, seq_length1 = get_data(
        r"train_data/主机电流样本.csv")
    trainX2, trainY, testX2, testY, seq_length2 = get_data(
        r"train_data/负压样本.csv")
    trainX3, trainY, testX3, testY, seq_length3 = get_data(
        r"train_data/料浆样本.csv")
    trainX4, trainY, testX4, testY, seq_length4 = get_data(
        r"train_data/喂煤样本.csv")
    trainX5, trainY, testX5, testY, seq_length5 = get_data(
        r"train_data/窑头温度样本.csv")
    trainX6, trainY, testX6, testY, seq_length6 = get_data(
        r"train_data/窑尾温度样本.csv")
    trainX7, trainY, testX7, testY, seq_length7 = get_data(
        r"train_data/一次风样本.csv")
    seq_lengths = [seq_length1, seq_length2, seq_length3,
                   seq_length4, seq_length5, seq_length6, seq_length7]
    model = SequenceClassify(129, 3, 10, seq_lengths)
    # print(model.__dict__)
    # print(model['feature_layer1'])
    # print(model['feature_layer7'])

    # for name, p in model.named_parameters():
    #     print("name:{}".format(name))

    valid_data = SequenceData(testX1, testX2, testX3,
                              testX4, testX5, testX6, testX7, testY)

    valid_dataloader = DataLoader(dataset=valid_data, batch_size=len(
        valid_data), shuffle=False, num_workers=0, drop_last=False)

    utils.load_checkpoint(r"./model/best.pth.tar", model)
    model.eval()

    for batch in valid_dataloader:
        x1, x2, x3, x4, x5, x6, x7, y = batch
        f1 = model.feature_layer1(x1).squeeze(2).detach().numpy()
        f2 = model.feature_layer2(x2).squeeze(2).detach().numpy()
        f3 = model.feature_layer3(x3).squeeze(2).detach().numpy()
        f4 = model.feature_layer4(x4).squeeze(2).detach().numpy()
        f5 = model.feature_layer5(x5).squeeze(2).detach().numpy()
        f6 = model.feature_layer6(x6).squeeze(2).detach().numpy()
        f7 = model.feature_layer7(x7).squeeze(2).detach().numpy()
        y = y.detach().numpy()
        show_feature(x1, y, "主机电流_init.png")
        show_feature(f1, y, "主机电流_cnn.png")
        show_feature(x2, y, "负压样本_init.png")
        show_feature(f2, y, "负压样本_cnn.png")
        show_feature(x3, y, "料浆样本_init.png")
        show_feature(f3, y, "料浆样本_cnn.png")
        show_feature(x4, y, "喂煤样本_init.png")
        show_feature(f4, y, "喂煤样本_cnn.png")
        show_feature(x5, y, "窑头温度样本_init.png")
        show_feature(f5, y, "窑头温度样本_cnn.png")
        show_feature(x6, y, "窑尾温度样本_init.png")
        show_feature(f6, y, "窑尾温度样本_cnn.png")
        show_feature(x7, y, "一次风样本_init.png")
        show_feature(f7, y, "一次风样本_cnn.png")
