import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import log_loss
import gc
from scipy import sparse
from sklearn.model_selection import train_test_split
from utils import set_logger, setup_seed
import logging
import warnings
warnings.filterwarnings('ignore')
setup_seed(2021)

"""
ref:https://blog.csdn.net/u013074302/article/details/76419592

"""

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


def gbdt_select_features(trainX, trainY, validX, validY,prefix='gbdt_leaf_'):
    logging.info("开始训练树模型...")
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample=0.8,
                            min_child_weight=0.5,
                            colsample_bytree=0.7,
                            num_leaves=100,
                            max_depth=12,
                            learning_rate=0.05,
                            n_estimators=10,
                            )

    gbm.fit(trainX, trainY,
            eval_set=[(trainX, trainY), (validX, validY)],
            eval_names=['train', 'val'],
            eval_metric='binary_logloss',
            early_stopping_rounds=100
            )
    model = gbm.booster_
    logging.info('训练得到叶子数')
    gbdt_feats_train = model.predict(trainX, pred_leaf=True)
    gbdt_feats_valid = model.predict(validX, pred_leaf=True)
    gbdt_feats_name = [prefix + str(i)
                       for i in range(gbdt_feats_train.shape[1])]
    df_train_gbdt_feats = pd.DataFrame(
        gbdt_feats_train, columns=gbdt_feats_name)
    df_test_gbdt_feats = pd.DataFrame(
        gbdt_feats_valid, columns=gbdt_feats_name)
    return df_train_gbdt_feats, df_test_gbdt_feats, gbdt_feats_name


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

    def gbdt_lr_cvr_task1(self):
        """
        用户ID类特征和广告ID类特征构建树模型:
        """
        # 加载数据
        dataSet = pd.read_csv(self.csvfile, sep='|')
        # 归一化
        logging.info('开始对连续特征归一化...')
        scaler = MinMaxScaler()
        for col in self.continue_feature_names:
            dataSet[col] = scaler.fit_transform(
                dataSet[col].values.reshape(-1, 1))

        # trainX, trainY, testX, testY

        trainX_continue_features, trainY, validX_continue_features, validY = split_dataSet(
            dataSet[self.continue_feature_names], dataSet[self.target])

        logging.info('归一化结束')
        # one-hot 编码
        user_id_features = []
        logging.info('开始对user-ID类特征进行one-hot编码...')
        for col in self.user_id_feature_names:
            onehot_feats = pd.get_dummies(dataSet[col], prefix=col)
            user_id_features.extend(list(onehot_feats.columns))
            dataSet.drop([col], axis=1, inplace=True)
            dataSet = pd.concat([dataSet, onehot_feats], axis=1)
        logging.info('user-ID类特征one-hot编码结束...')
        trainX, trainY, testX, testY = split_dataSet(
            dataSet[user_id_features], dataSet[self.target])
        train_user_gbdt_feats, valid_user_gbdt_feats, user_gbdt_feats_name = gbdt_select_features(
            trainX, trainY, testX, testY,prefix='gbdt_user_leaf_')
        logging.info("gbdt 对user-id 选择的特征:{}".format(user_gbdt_feats_name))
        logging.info('开始对ad-ID类特征进行one-hot编码...')
        ad_id_features = []
        for col in self.ad_id_feature_names:
            onehot_feats = pd.get_dummies(dataSet[col], prefix=col)
            ad_id_features.extend(onehot_feats.columns)
            dataSet.drop([col], axis=1, inplace=True)
            dataSet = pd.concat([dataSet, onehot_feats], axis=1)
        logging.info('ad-ID类特征one-hot编码结束...')
        trainX, trainY, testX, testY = split_dataSet(
            dataSet[ad_id_features], dataSet[self.target])
        train_ad_gbdt_feats, valid_ad_gbdt_feats, ad_gbdt_feats_name = gbdt_select_features(
            trainX, trainY, testX, testY,prefix='gbdt_ad_leaf_')
        logging.info("gbdt 对user-id 选择的特征:{}".format(ad_gbdt_feats_name))
        train_features = pd.concat(
            [trainX_continue_features, train_user_gbdt_feats, train_ad_gbdt_feats])
        valid_features = pd.concat(
            [validX_continue_features, valid_user_gbdt_feats, valid_ad_gbdt_feats])
        lr = LogisticRegression()
        lr.fit(train_features, trainY)
        tr_logloss = log_loss(trainY, lr.predict_proba(train_features)[:, 1])
        logging.info('tr-logloss: ', tr_logloss)
        val_logloss = log_loss(validY, lr.predict_proba(valid_features)[:, 1])
        logging.info('val-logloss: ', val_logloss)

    def gbdt_lr_cvr_task2(self):
        """
        使用稀疏类特征和密集类特征构建树模型:
        """
        # 加载数据
        dataSet = pd.read_csv(self.csvfile, sep='|')
        # 归一化
        logging.info('开始对连续特征归一化...')
        scaler = MinMaxScaler()
        for col in self.continue_feature_names:
            dataSet[col] = scaler.fit_transform(
                dataSet[col].values.reshape(-1, 1))

        # trainX, trainY, testX, testY

        trainX_continue_features, trainY, validX_continue_features, validY = split_dataSet(
            dataSet[self.continue_feature_names], dataSet[self.target])

        logging.info('归一化结束')
        # one-hot 编码
        logging.info('开始对稀疏类特征进行one-hot编码...')
        sparse_features = []
        for col in self.sparse_feature_names:
            onehot_feats = pd.get_dummies(dataSet[col], prefix=col)
            sparse_features.extend(onehot_feats.columns)
            dataSet.drop([col], axis=1, inplace=True)
            dataSet = pd.concat([dataSet, onehot_feats], axis=1)
        logging.info('对稀疏类特征进行one-hot编码结束...')
        trainX, trainY, testX, testY = split_dataSet(
            dataSet[sparse_features], dataSet[self.target])
        train_sparse_gbdt_feats, valid_sparse_gbdt_feats, sparse_gbdt_feats_name = gbdt_select_features(
            trainX, trainY, testX, testY,prefix='gbdt_sparse_leaf_')
        logging.info("gbdt 对稀疏类特征树选择的特征:{}".format(sparse_gbdt_feats_name))

        logging.info('开始对密集类特征进行one-hot编码...')
        dense_features = []
        for col in self.dense_feature_names:
            onehot_feats = pd.get_dummies(dataSet[col], prefix=col)
            dense_features.extend(onehot_feats.columns)
            dataSet.drop([col], axis=1, inplace=True)
            dataSet = pd.concat([dataSet, onehot_feats], axis=1)
        logging.info('密集类特征one-hot编码结束...')
        trainX, trainY, testX, testY = split_dataSet(
            dataSet[dense_features], dataSet[self.target])
        train_dense_gbdt_feats, valid_dense_gbdt_feats, dense_gbdt_feats_name = gbdt_select_features(
            trainX, trainY, testX, testY,prefix='gbdt_dense_leaf_')
        logging.info("gbdt 对密集类选择的特征:{}".format(dense_gbdt_feats_name))

        train_features = pd.concat(
            [trainX_continue_features, train_sparse_gbdt_feats, train_dense_gbdt_feats])
        valid_features = pd.concat(
            [validX_continue_features, valid_sparse_gbdt_feats, valid_dense_gbdt_feats])
        lr = LogisticRegression()
        lr.fit(train_features, trainY)
        tr_logloss = log_loss(trainY, lr.predict_proba(train_features)[:, 1])
        logging.info('tr-logloss: ', tr_logloss)
        val_logloss = log_loss(validY, lr.predict_proba(valid_features)[:, 1])
        logging.info('val-logloss: ', val_logloss)


if __name__ == "__main__":
    job = CVRJob()
    job.gbdt_lr_cvr_task1()
    job.gbdt_lr_cvr_task2()
