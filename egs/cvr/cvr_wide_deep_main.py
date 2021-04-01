import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss,roc_curve,auc,roc_auc_score
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM,WDL,xDeepFM
import gc
from scipy import sparse
from sklearn.model_selection import train_test_split
from utils import set_logger, setup_seed, get_device
import logging
import warnings
import math
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings('ignore')
setup_seed(2021)



def draw_figure(train_log,valid_log,png_file,variable='logloss'):
    fig1 = plt.figure()
    plt.plot(range(1,len(train_log)+1),train_log,label="Training {}".format(variable))
    plt.plot(range(1,len(valid_log)+1),valid_log,label="Validation {}".format(variable))
    plt.ylim(0,1)
    plt.xlim(0,len(train_log)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig1.savefig(png_file,bbox_inches='tight',dpi=300)

def draw_loss(loss_fold):
    models = ['deepfm','wdl','xdeepfm']
    for model in models:
        xlsfile = os.path.join(loss_fold,"train_val_log_{}.xlsx".format(model))
        if os.path.exists(xlsfile) == False:
            continue
        data = pd.read_excel(xlsfile)
        train_loss = data['logloss']
        valid_loss = data['val_logloss']
        png_file = os.path.join(loss_fold,"{}_logloss.png".format(model))
        draw_figure(train_loss,valid_loss,png_file,variable='logloss')

        train_auc = data['auc']
        valid_auc = data['val_auc']
        png_file = os.path.join(loss_fold,"{}_auc.png".format(model))
        draw_figure(train_auc,valid_auc,png_file,variable='auc')


project_config = {
    'csvfile': "train_data_final2.csv",
    'label': "consume_purchase",
    # 用户类特征
    'user_id_features': ['city', 'city_rank', 'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence', 'emui_dev'],
    'ad_id_features': ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'indu_name'],  # 广告类特征
    # 连续类特征
    'continue_features': ['age', 'app_score', 'list_time', 'device_price', 'communication_onlinerate', 'communication_avgonline_30d'],
    # 稀疏类特征
    'sparse_features': ['city', 'communication_onlinerate', 'task_id', 'adv_id'],
    'dense_features': ['city_rank', 'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence', 'emui_dev', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'indu_name']  # 密集类特征
}


def split_dataSet(inputX, target, test_size=0.2):
    trainX, testX, trainY, testY = train_test_split(
        inputX, target, test_size=test_size, random_state=0)
    return trainX, trainY, testX, testY

def eval_log_loss(y_true,y_pred):
    summ = 0.0
    for y1,y2 in zip(y_true,y_pred):
        summ -= (y1*math.log(y2)+(1-y1)*math.log(1-y2))
    return summ / len(y_true)


class CVRJob(object):
    def __init__(self, project_config=project_config):
        self.target = project_config['label']
        self.sparse_feature_names = project_config['sparse_features']
        self.dense_feature_names = project_config['dense_features']
        self.user_id_feature_names = project_config['user_id_features']
        self.ad_id_feature_names = project_config['ad_id_features']
        self.continue_feature_names = project_config['continue_features']
        self.csvfile = project_config['csvfile']
        # self.dataSet = pd.read_csv(self.csvfile, sep='|')
        # self.trainX, self.validX, self.trainY, self.validY = train_test_split(self.dataSet.drop(
        #     ['label', 'consume_purchase'], axis=1), self.dataSet[self.target], test_size=0.2, random_state=0)
        self.log_file = set_logger("./train.log")

    def start_work(self):
        data = pd.read_csv(self.csvfile,sep='|')
        sparse_features = self.user_id_feature_names + self.ad_id_feature_names
        dense_features = self.continue_feature_names


        target = [self.target]
        data[sparse_features] = data[sparse_features].fillna('-1',)
        data[dense_features] = data[dense_features].fillna(0,)

        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])
        mms = MinMaxScaler(feature_range=(0,1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        fixlen_feature_columns = [SparseFeat(feat,data[feat].nunique()) for feat in sparse_features] + [DenseFeat(feat,1,) for feat in dense_features]

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        train, test = train_test_split(data,test_size=0.1,random_state=2021)

        train_model_input = {name:train[name] for name in feature_names}
        test_model_input = {name:test[name] for name in feature_names}

        device = get_device()

        model = xDeepFM(linear_feature_columns=linear_feature_columns,dnn_feature_columns = dnn_feature_columns,task='binary',l2_reg_embedding=1e-4,device=device)

        model.compile("adagrad","binary_crossentropy",metrics=['logloss','auc'])

        history, train_logs = model.fit(train_model_input,train[target].values,batch_size=1024,epochs=20,validation_split=0.1,verbose=2)

        pred_ans = model.predict(test_model_input,1024)

        df = pd.DataFrame(data=train_logs)
        df.to_excel("./result/train_val_log_xdeepfm.xlsx")

        print("test LogLoss: ",round(log_loss(test[target].values,pred_ans),4))
        print("test AUC: ",round(roc_auc_score(test[target].values,pred_ans),4))


if __name__ == "__main__":
    job = CVRJob()
    job.start_work()
    # draw_loss("./result")
