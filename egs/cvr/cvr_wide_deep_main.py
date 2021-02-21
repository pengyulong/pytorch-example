import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss,roc_curve,auc
import gc
from scipy import sparse
from sklearn.model_selection import train_test_split
from utils import set_logger, setup_seed
import logging
import warnings
warnings.filterwarnings('ignore')
setup_seed(2021)

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

    def train(self):
        pass

    