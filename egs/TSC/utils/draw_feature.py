from sklearn.manifold import TSNE
from attention_dpcnn import SequenceClassify
from matplotlib import cm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


def plot_with_labels(lowDWeights, labels,pngfile):
    plt.cla()
    # 降到二维了，分别给x和y
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    # 遍历每个点以及对应标签
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255/9 * s)) 
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        # plt.text(x,y,s,color=cm.Set1(s/3.),fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max()); 
    plt.title('Visualize init input layer')

    plt.savefig("{}.jpg".format(pngfile),dpi=300)

def process_data(csvfile):
    dataSet = pd.read_csv(csvfile,index_col=0,header=None)
    dataX = np.array(dataSet.iloc[:,:-1])
    dataY = np.array(dataSet.iloc[:,-1]-1)
    trainX,testX,trainY,testY = train_test_split(dataX,dataY,test_size=0.2,random_state=0)
    seq_length = dataX.shape[1]
    dataX = normalize(dataX,axis=1,norm='max')
    return testX,testY,seq_length

def plot_feature(dataX,dataY,pngfile,seq_length=None):
    tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    nsamples = len(dataY) #显示500个样本
    low_dim_embs = tsne.fit_transform(dataX[:nsamples,:])
    labels = dataY[:nsamples]
    plot_with_labels(low_dim_embs,labels,pngfile)

if __name__ == "__main__":
    # csvfile1 = r"train_data/主机电流样本.csv"
    # csvfile2 = r"train_data/负压样本.csv"
    # csvfile3 = r"train_data/料浆样本.csv"
    # csvfile4 = r"train_data/喂煤样本.csv"
    # csvfile5 = r"train_data/窑头温度样本.csv"
    # csvfile6 = r"train_data/窑尾温度样本.csv"
    # csvfile7 = r"train_data/一次风样本.csv"
    data_root = r"train_data"
    imge_root = r"image_original"
    if os.path.exists(imge_root)==False:
        os.mkdir(imge_root)
    for csvfile in os.listdir(data_root):
        pngfile = os.path.join(imge_root,csvfile[0:-4])
        csvfile = os.path.join(data_root,csvfile)
        dataX,dataY,seq_length = process_data(csvfile)
        plot_feature(dataX,dataY,pngfile)
        



