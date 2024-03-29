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
from time import time
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
    # 广告类特征
    'ad_id_features': ['task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'indu_name'],  # 广告类特征
    # 数值类特征
    'continue_features': ['age', 'app_score', 'list_time', 'device_price', 'communication_onlinerate', 'communication_avgonline_30d'],
    # 稀疏类特征
    'sparse_features': ['city', 'communication_onlinerate', 'task_id', 'adv_id'],
    # 密集类特征
    'dense_features': ['city_rank', 'device_name', 'device_size', 'career', 'gender', 'net_type', 'residence', 'emui_dev', 'creat_type_cd', 'adv_prim_id', 'dev_id', 'inter_type_cd', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class', 'indu_name']
}


def split_dataSet(inputX, target, test_size=0.2):
    trainX, testX, trainY, testY = train_test_split(
        inputX, target, test_size=test_size, random_state=0)
    return trainX, trainY, testX, testY


def gbdt_select_features(trainX, trainY, validX, validY, n_estimators = 10, prefix='gbdt_leaf_'):
    logging.info("开始训练树模型...")
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample=0.8,
                            min_child_weight=0.5,
                            colsample_bytree=0.7,
                            num_leaves=100,
                            max_depth=12,
                            learning_rate=0.05,
                            n_estimators=n_estimators,
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
        gbdt_feats_train, columns=gbdt_feats_name,index=trainX.index)
    df_test_gbdt_feats = pd.DataFrame(
        gbdt_feats_valid, columns=gbdt_feats_name,index=validX.index)
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

    def gbdt_cvr(self):
        # 加载数据
        t1 = time()
        dataSet = pd.read_csv(self.csvfile, sep='|')
        dataSet['id'] = range(1,len(dataSet)+1)
        dataSet.set_index('id',inplace=True)
        id_features = self.user_id_feature_names + self.ad_id_feature_names
        id_onehot_features = []
        # 归一化
        logging.info('开始对连续特征归一化...')
        scaler = MinMaxScaler()
        for col in self.continue_feature_names:
            dataSet[col] = scaler.fit_transform(
                dataSet[col].values.reshape(-1, 1))
        trainX_continue_features, trainY, validX_continue_features, validY = split_dataSet(
            dataSet[self.continue_feature_names], dataSet[self.target])
        logging.info('归一化结束')
        # one-hot 编码
        user_id_features = []
        logging.info('开始对ID类特征进行one-hot编码...')
        for col in id_features:
            onehot_feats = pd.get_dummies(dataSet[col], prefix=col)
            id_onehot_features.extend(list(onehot_feats.columns))
            user_id_features.extend(list(onehot_feats.columns))
            # id_features.extend(list(onehot_feats.columns))
            dataSet.drop([col], axis=1, inplace=True)
            dataSet = pd.concat([dataSet, onehot_feats], axis=1,join_axes=[dataSet.index])
        logging.info('ID类特征one-hot编码结束...')
        trainX, trainY, validX, validY = split_dataSet(
            dataSet[user_id_features], dataSet[self.target])
        train_id_gbdt_feats, valid_id_gbdt_feats, id_gbdt_feats_name = gbdt_select_features(
            trainX, trainY, validX, validY,n_estimators=20,prefix='gbdt_id_leaf_')
        logging.info("gbdt 对id类选择的特征:{}".format(id_gbdt_feats_name))
        train_features = pd.concat(
            [trainX_continue_features, train_id_gbdt_feats],axis=1,join_axes=[trainX_continue_features.index])
        valid_features = pd.concat(
            [validX_continue_features, valid_id_gbdt_feats],axis=1,join_axes=[validX_continue_features.index])
        lr = LogisticRegression()
        lr.fit(train_features, trainY)
        t2 = time()
        logging.info("task:单独为id类特征构建gbdt树然后与连续类特征进行lr对cvr进行预估")
        tr_logloss = log_loss(trainY,lr.predict_proba(train_features))
        trainY_pred = lr.predict_proba(train_features)
        fpr, tpr, thresholds = roc_curve(trainY,trainY_pred[:,1])
        logging.info('tr-logloss:{},auc:{}'.format(tr_logloss,auc(fpr,tpr)))
        # val_logloss = log_loss(validY, lr.predict_proba(valid_features)[:, 1])
        val_logloss = log_loss(validY,lr.predict_proba(valid_features))
        validY_pred = lr.predict_proba(valid_features)
        fpr, tpr, thresholds = roc_curve(validY,validY_pred[:,1])
        logging.info('val-logloss:{},auc:{}'.format(val_logloss,auc(fpr,tpr)))
        logging.info("单独对id类特征进行gbdt组合成新的特征然后连续类特征进行lr预估cvr:{}s".format(t2-t1))
        

    def lr_cvr(self):
        # 加载数据
        dataSet = pd.read_csv(self.csvfile, sep='|')
        dataSet['id'] = range(1,len(dataSet)+1)
        dataSet.set_index('id',inplace=True)
        id_features = self.user_id_feature_names + self.ad_id_feature_names
        id_onehot_features = []
        # 归一化
        logging.info('开始对连续特征归一化...')
        scaler = MinMaxScaler()
        for col in self.continue_feature_names:
            dataSet[col] = scaler.fit_transform(
                dataSet[col].values.reshape(-1, 1))
        trainX_continue_features, trainY, validX_continue_features, validY = split_dataSet(
            dataSet[self.continue_feature_names], dataSet[self.target])
        logging.info('归一化结束')
        # one-hot 编码
        user_id_features = []
        logging.info('开始对ID类特征进行one-hot编码...')
        for col in id_features:
            onehot_feats = pd.get_dummies(dataSet[col], prefix=col)
            id_onehot_features.extend(list(onehot_feats.columns))
            # id_features.extend(list(onehot_feats.columns))
            dataSet.drop([col], axis=1, inplace=True)
            dataSet = pd.concat([dataSet, onehot_feats], axis=1,join_axes=[dataSet.index])
        logging.info('ID类特征one-hot编码结束...')
        trainX, trainY, validX, validY = split_dataSet(
            dataSet[id_onehot_features], dataSet[self.target])
        train_features = pd.concat([trainX,trainX_continue_features],axis=1,join_axes=[trainX.index])
        valid_features = pd.concat([validX,validX_continue_features],axis=1,join_axes=[validX.index])
        lr = LogisticRegression()
        lr.fit(train_features, trainY)
        train_pred = lr.predict_proba(trainX)
        valid_pred = lr.predict_proba(validX)
        logging.info("baseline: lr建模准确性能指标")
        tr_logloss = log_loss(trainY,train_pred)
        fpr, tpr, thresholds = roc_curve(trainY,train_pred[:,1])
        logging.info('train-logloss:{},auc:{}'.format(tr_logloss,auc(fpr,tpr)))
        val_logloss = log_loss(validY,valid_pred)
        fpr, tpr, thresholds = roc_curve(validY,valid_pred[:,1])
        logging.info('valid-logloss:{},auc:{}'.format(val_logloss,auc(fpr,tpr)))

    def gbdt_lr_cvr_task1(self):
        """
        用户ID类特征和广告ID类特征构建树模型:
        """
        # 加载数据
        t1 = time()
        dataSet = pd.read_csv(self.csvfile, sep='|')
        dataSet['id'] = range(1,len(dataSet)+1)
        dataSet.set_index('id',inplace=True)
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
            dataSet = pd.concat([dataSet, onehot_feats], axis=1,join_axes=[dataSet.index])
        logging.info('user-ID类特征one-hot编码结束...')
        trainX, trainY, testX, testY = split_dataSet(
            dataSet[user_id_features], dataSet[self.target])
        train_user_gbdt_feats, valid_user_gbdt_feats, user_gbdt_feats_name = gbdt_select_features(
            trainX, trainY, testX, testY,prefix='gbdt_user_leaf_')
        # print("train_user_gbdt_feats:{}".format(train_user_gbdt_feats))
        logging.info("gbdt 对user-id 选择的特征:{}".format(user_gbdt_feats_name))
        logging.info('开始对ad-ID类特征进行one-hot编码...')
        ad_id_features = []
        for col in self.ad_id_feature_names:
            onehot_feats = pd.get_dummies(dataSet[col], prefix=col)
            ad_id_features.extend(onehot_feats.columns)
            dataSet.drop([col], axis=1, inplace=True)
            dataSet = pd.concat([dataSet, onehot_feats], axis=1,join_axes=[dataSet.index])
        logging.info('ad-ID类特征one-hot编码结束...')
        trainX, trainY, testX, testY = split_dataSet(
            dataSet[ad_id_features], dataSet[self.target])
        train_ad_gbdt_feats, valid_ad_gbdt_feats, ad_gbdt_feats_name = gbdt_select_features(
            trainX, trainY, testX, testY,prefix='gbdt_ad_leaf_')
        logging.info("gbdt 对user-id 选择的特征:{}".format(ad_gbdt_feats_name))
        logging.info("训练集个数:{},测试集个数:{}".format(len(trainY),len(testY)))
        train_features = pd.concat(
            [trainX_continue_features, train_user_gbdt_feats, train_ad_gbdt_feats],axis=1,join_axes=[trainX_continue_features.index])
        valid_features = pd.concat(
            [validX_continue_features, valid_user_gbdt_feats, valid_ad_gbdt_feats],axis=1,join_axes=[validX_continue_features.index])
        lr = LogisticRegression()
        lr.fit(train_features, trainY)
        t2 = time()
        logging.info("task:分别构建user-id特征数和ad-id特征数的的性能指标")
        tr_logloss = log_loss(trainY,lr.predict_proba(train_features))
        trainY_pred = lr.predict_proba(train_features)
        fpr, tpr, thresholds = roc_curve(trainY,trainY_pred[:,1])
        logging.info('tr-logloss:{},auc:{}'.format(tr_logloss,auc(fpr,tpr)))
        # val_logloss = log_loss(validY, lr.predict_proba(valid_features)[:, 1])
        val_logloss = log_loss(validY,lr.predict_proba(valid_features))
        validY_pred = lr.predict_proba(valid_features)
        fpr, tpr, thresholds = roc_curve(validY,validY_pred[:,1])
        logging.info('val-logloss:{},auc:{}'.format(val_logloss,auc(fpr,tpr)))
        logging.info("分别构建user-id和ad-id然后与连续类特征使用lr对cvr进行预估所花费的时间:{}s".format(t2-t1))

    def gbdt_lr_cvr_task2(self):
        """
        使用稀疏类特征和密集类特征构建树模型:
        """
        # 加载数据
        t1 = time()
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
            [trainX_continue_features, train_sparse_gbdt_feats, train_dense_gbdt_feats],axis=1,join_axes=[trainX_continue_features.index])
        valid_features = pd.concat(
            [validX_continue_features, valid_sparse_gbdt_feats, valid_dense_gbdt_feats],axis=1,join_axes=[validX_continue_features.index])
        lr = LogisticRegression()
        lr.fit(train_features, trainY)
        t2 = time()
        logging.info("task:分别构建稀疏类特征数和密集类特征数的的性能指标")
        # tr_logloss = log_loss(trainY, lr.predict_proba(train_features)[:, 1])
        tr_logloss = log_loss(trainY,lr.predict_proba(train_features))
        trainY_pred = lr.predict_proba(train_features)
        fpr, tpr, thresholds = roc_curve(trainY,trainY_pred[:,1])
        logging.info('tr-logloss:{},auc:{}'.format(tr_logloss,auc(fpr,tpr)))
        # val_logloss = log_loss(validY, lr.predict_proba(valid_features)[:, 1])
        val_logloss = log_loss(validY,lr.predict_proba(valid_features))
        validY_pred = lr.predict_proba(valid_features)
        fpr, tpr, thresholds = roc_curve(validY,validY_pred[:,1])
        logging.info('val-logloss:{},auc:{}'.format(val_logloss,auc(fpr,tpr)))
        logging.info("使用稀疏类特征和密集类特征进行gbdt筛选特征,然后与连续类特征使用lr对cvr进行预估所耗费的时间:{}s".format(t2-t1))


if __name__ == "__main__":
    job = CVRJob()
    job.lr_cvr()
    #job.gbdt_cvr()
    #job.gbdt_lr_cvr_task1()
    #job.gbdt_lr_cvr_task2()
