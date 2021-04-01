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

def gbdt_select_features(trainX, trainY, validX, validY, n_estimators = 10, prefix='gbdt_leaf_'):
    logging.info("开始训练树模型...")
    gbm = lgb.LGBMRegressor(objective='binary',
                            subsample=0.8,
                            min_child_weight=0.5,
                            colsample_bytree=0.7,
                            num_leaves=100,
                            max_depth=12,
                            learning_rate=0.1,
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

def draw_figure(train_log,valid_log,png_file,min_y,max_y,variable='logloss'):
    fig1 = plt.figure()
    plt.plot(range(1,len(train_log)+1),train_log,label="Training {}".format(variable))
    plt.plot(range(1,len(valid_log)+1),valid_log,label="Validation {}".format(variable))
    plt.ylim(min_y,max_y)
    plt.xlim(0,len(train_log)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.xlabel("Epochs ")
    plt.ylabel(variable)
    fig1.savefig(png_file,bbox_inches='tight',dpi=300)

def draw_exp_compare(loss_fold):
    fig = plt.figure()
    layer_num = [2,3,4]
    auc = [0.6916,0.6916,0.6918]
    logloss = [0.4010,0.4010,0.4009]
    fig1 = plt.figure()
    plt.plot(layer_num,logloss,'*k')
    plt.ylim(0.3900,0.4020)
    fig1.savefig("test.png")
    plt.show()


def draw_loss(loss_fold):
    # models = ['deepfm','wdl','xdeepfm','mix']
    models = ['ex1','ex2','ex3','ex4','ex5','ex6','ex7','ex8','ex9']
    for model in models:
        xlsfile = os.path.join(loss_fold,"train_val_log_{}.xlsx".format(model))
        if os.path.exists(xlsfile) == False:
            continue
        data = pd.read_excel(xlsfile)
        train_loss = data['logloss']
        valid_loss = data['val_logloss']
        png_file = os.path.join(loss_fold,"mix_{}_logloss.png".format(model))
        draw_figure(train_loss,valid_loss,png_file,0.38,0.42,variable='logloss')

        train_auc = data['auc']
        valid_auc = data['val_auc']
        png_file = os.path.join(loss_fold,"mix_{}_auc.png".format(model))
        draw_figure(train_auc,valid_auc,png_file,0.67,0.72,variable='auc')


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
        self.n_estimators = 70
        self.dnn_hidden_units = (128,512,256)
        self.epochs = 20
        self.act_func = 'sigmoid'
        self.result_xlsfile = "./result/train_val_log_ex20.xlsx"
        # self.dataSet = pd.read_csv(self.csvfile, sep='|')
        # self.trainX, self.validX, self.trainY, self.validY = train_test_split(self.dataSet.drop(
        #     ['label', 'consume_purchase'], axis=1), self.dataSet[self.target], test_size=0.2, random_state=0)
        self.log_file = set_logger("./train.log")

    def start_single_work(self):
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

        X_train, X_test, y_train, y_test = train_test_split(data[feature_names],data[target], test_size=0.2, random_state=2021)

        train_model_input = {name:X_train[name] for name in feature_names}
        test_model_input = {name:X_test[name] for name in feature_names}

        device = get_device()

        model = WDL(linear_feature_columns=linear_feature_columns,dnn_feature_columns = dnn_feature_columns,task='binary',l2_reg_embedding=5e-4,device=device,dnn_hidden_units=self.dnn_hidden_units,dnn_activation='relu')

        model.compile("adagrad","binary_crossentropy",metrics=['logloss','auc'])

        history, train_logs = model.fit(train_model_input,y_train.values,batch_size=1024,epochs=self.epochs,validation_data=(test_model_input,y_test.values),verbose=2)

        pred_ans = model.predict(test_model_input,1024)

        df = pd.DataFrame(data=train_logs)
        df.to_excel(self.result_xlsfile)

        print("test LogLoss: ",round(log_loss(y_test.values,pred_ans),4))
        print("test AUC: ",round(roc_auc_score(y_test.values,pred_ans),4))

    def start_work2(self):
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

        feature_names = sparse_features + dense_features
        X_train, X_test, y_train, y_test = train_test_split(data[feature_names],data[target], test_size=0.1, random_state=2021)
        # logging.info("X_train's columns:{}".format(list(X_train.columns())))

        train_gbdt_user_id_feats, test_gbdt_user_id_feats, gbdt_user_id_feats_name = gbdt_select_features(X_train[feature_names],y_train,X_test[feature_names],y_test,n_estimators=self.n_estimators,prefix='gbdt_leaf_user_id_')
        logging.info("gbdt_feats_user_id_name:{}".format(gbdt_user_id_feats_name))
        train_gbdt_ad_id_feats, test_gbdt_ad_id_feats, gbdt_ad_id_feats_name = gbdt_select_features(X_train[feature_names],y_train,X_test[feature_names],y_test,n_estimators=self.n_estimators,prefix='gbdt_leaf_ad_id_')
        logging.info("gbdt_feats_ad_id_name:{}".format(gbdt_ad_id_feats_name))

        X_train = pd.concat([X_train, train_gbdt_user_id_feats,train_gbdt_ad_id_feats], axis=1,join_axes=[X_train.index])
        X_test = pd.concat([X_test, test_gbdt_user_id_feats,test_gbdt_ad_id_feats], axis=1,join_axes=[X_test.index])

        linear_feature_columns = fixlen_feature_columns + [SparseFeat(feat,len(set(X_train[feat].tolist() + X_test[feat].tolist()))) for feat in gbdt_ad_id_feats_name+gbdt_user_id_feats_name+sparse_features]
        dnn_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        logging.info("feature_names:{}".format(feature_names))
        train_model_input = {name:X_train[name] for name in feature_names}
        test_model_input = {name:X_test[name] for name in feature_names}

        device = get_device()

        model = WDL(linear_feature_columns=linear_feature_columns,dnn_feature_columns = dnn_feature_columns,task='binary',l2_reg_embedding=5e-4,device=device,dnn_hidden_units=self.dnn_hidden_units,dnn_activation='relu')

        model.compile("adagrad","binary_crossentropy",metrics=['logloss','auc'])

        # logging.info("y_train.values:{}".format(y_train.values))
        # logging.info("y_test's value:{}".format(y_test.values))

        history, train_logs = model.fit(train_model_input,y_train.values,batch_size=1024,epochs=self.epochs,validation_data=(test_model_input,y_test.values),verbose=2)

        pred_ans = model.predict(test_model_input,1024)

        df = pd.DataFrame(data=train_logs)
        df.to_excel(self.result_xlsfile)

        print("test LogLoss: ",round(log_loss(y_test.values,pred_ans),4))
        print("test AUC: ",round(roc_auc_score(y_test.values,pred_ans),4))

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

        feature_names = sparse_features + dense_features
        X_train, X_test, y_train, y_test = train_test_split(data[feature_names],data[target], test_size=0.2, random_state=2021)
        # logging.info("X_train's columns:{}".format(list(X_train.columns())))

        train_gbdt_feats, test_gbdt_feats, gbdt_feats_name = gbdt_select_features(X_train[feature_names],y_train,X_test[feature_names],y_test,n_estimators=self.n_estimators,prefix='gbdt_leaf_feats_')
        
        X_train = pd.concat([X_train, train_gbdt_feats], axis=1,join_axes=[X_train.index])
        X_test = pd.concat([X_test, test_gbdt_feats], axis=1,join_axes=[X_test.index])

        linear_feature_columns = fixlen_feature_columns + [SparseFeat(feat,len(set(X_train[feat].tolist() + X_test[feat].tolist()))) for feat in gbdt_feats_name]
        
        dnn_feature_columns = linear_feature_columns
    

        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
        logging.info("feature_names:{}".format(feature_names))
        train_model_input = {name:X_train[name] for name in feature_names}
        test_model_input = {name:X_test[name] for name in feature_names}

        device = get_device()

        model = WDL(linear_feature_columns=linear_feature_columns,dnn_feature_columns = dnn_feature_columns,task='binary',l2_reg_embedding=5e-4,device=device,dnn_hidden_units=self.dnn_hidden_units,dnn_activation=self.act_func)

        model.compile("adagrad","binary_crossentropy",metrics=['logloss','auc'])

        history, train_logs = model.fit(train_model_input,y_train.values,batch_size=1024,epochs=self.epochs,validation_data=(test_model_input,y_test.values),verbose=2)

        pred_ans = model.predict(test_model_input,1024)

        df = pd.DataFrame(data=train_logs)
        df.to_excel(self.result_xlsfile)

        print("test LogLoss: ",round(log_loss(y_test.values,pred_ans),4))
        print("test AUC: ",round(roc_auc_score(y_test.values,pred_ans),4))
if __name__ == "__main__":
    job = CVRJob()
    job.start_work()
    # draw_loss("./result")
    # draw_exp_compare('./result')
