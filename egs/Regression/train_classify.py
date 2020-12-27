# -*- coding: utf-8 -*-
"""
Created on 2020/11/13
@author: pengyulong
"""
from __future__ import division, print_function
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def load_dataSet(dataSet,feature_index,target_label):
    lb_maker1 = LabelEncoder()
    lb_maker2 = LabelEncoder()
    dataSet['Temporal Distribution'] = lb_maker1.fit_transform(dataSet['Temporal Distribution'])
    dataSet['Spatial Distribution'] = lb_maker2.fit_transform(dataSet['Spatial Distribution'])
    target = np.array(dataSet[target_label])
    inputX = np.array(dataSet.loc[:,feature_index])
    trainX,testX,trainY,testY = train_test_split(inputX,target,test_size=0.2,random_state=0)
    return trainX,trainY,testX,testY

def regression_loss_func(y_true,y_pred):
    corr = metrics.r2_score(y_true,y_pred)
    # Rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    rae = np.sqrt(abs(y_true - y_pred).sum()/abs(y_true - y_true.mean()).sum())
    return corr/rae  #accu = metrics.accuracy_score(y_true, y_pred)

def train_classify(dataSet,feature_index,modelname,target_label,mode):
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    trainX,trainY,testX,testY = load_dataSet(dataSet,feature_index,target_label)
    # print("训练样本个数:%d,测试样本个数:%d"%(len(trainY),len(testY)))
    my_score = make_scorer(regression_loss_func,greater_is_better=True)
    sc = StandardScaler()
    reg,param_grid = None,None
    if mode == 'xgboost':
        reg = XGBRegressor()
        param_grid = [{'reg__n_estimators':[50,100,200,300],'reg__max_depth':[2,4,5,6,7,8]}]
    elif mode == 'lightgbm':
        reg = LGBMRegressor()
        param_grid = [{'reg__num_leaves':[32,64,128]}]
    pipe = Pipeline(steps=[('sc',sc),('reg',reg)])
    model = GridSearchCV(pipe, param_grid,cv=5,scoring=my_score,n_jobs=-1)
    model.fit(trainX,trainY)
    print(model.best_params_)
    result_total = pd.DataFrame(index=dataSet.index,data={'y_pred':model.predict(dataSet.loc[:,feature_index]),target_label:dataSet[target_label]})
    result_train = pd.DataFrame(index=range(len(trainY)),data={'y_pred':model.predict(trainX),target_label:trainY})
    result_test = pd.DataFrame(index=range(len(testY)),data={'y_pred':model.predict(testX),target_label:testY})
    return result_total,result_train,result_test

def evaluate_model(dataSet,target_label):
    y_true = np.array(dataSet[target_label])
    y_pred = np.array(dataSet['y_pred'])
    corr = np.sqrt(metrics.r2_score(y_true,y_pred))
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = abs(y_true - y_pred).mean()
    rae = np.sqrt(abs(y_true - y_pred).sum()/abs(y_true - y_true.mean()).sum())
    rrse = np.sqrt(((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum())
    print("r2:{},rmse:{},mae:{},rae:{},rrse:{}".format(corr,rmse,mae,rae,rrse))

def train_finally_model(xlsfile,feature_index,modelname,method,mode,target_label):
    dataSet = pd.read_excel(xlsfile)
    # print(dataSet.head())
    print("--------------------开始训练{}模型,预测{}--------------------------".format(mode,target_label))
    result_total,result_train,result_test = train_classify(dataSet,feature_index,modelname,target_label,mode)
    print("--用{}训练后在整体集上的性能指标--:".format(mode))
    evaluate_model(result_total,target_label)
    print("--用{}训练后在测试集上的性能指标--:".format(mode))
    evaluate_model(result_test,target_label)
    print("--用{}训练后在训练集上的性能指标--:".format(mode))
    evaluate_model(result_train,target_label)
    
def train_lightgbm():
    import lightgbm as lgb
    params = {
        # 'object':'regression',
        'max_bin': 200,
        'learning_rate': 0.001,
        'num_leaves': 15,
        'metric':['l1','l2','rmse']
    }
    dataSet = pd.read_excel("data2.xlsx")
    trainX,trainY,testX,testY = load_dataSet(dataSet,feature_index=['Node Number','Thread Number','T/R','Spatial Distribution','Temporal Distribution'] ,target_label='Input Waiting Time')
    lgb_train = lgb.Dataset(trainX,trainY)
    lgb_eval = lgb.Dataset(testX,testY, reference=lgb_train)

    model = lgb.train(
        params, lgb_train,
        valid_sets=[lgb_train, lgb_eval],
        verbose_eval=10,
        num_boost_round=400,
        early_stopping_rounds=100
    )
    return model


def main():
    classify_index=['Node Number','Thread Number','T/R','Spatial Distribution','Temporal Distribution'] 
    method='Regressor'
    mode='lightgbm'
    xlsfilename="data2.xlsx"
    model_name = "classify"
    # target_label = 'Processor Utilization'
    # target_label = 'Channel Waiting Time'
    target_label = 'Input Waiting Time'
    # target_label = 'Network Response Time'
    # target_label = 'Channel Utilization'
    train_finally_model(xlsfilename,classify_index,model_name,method,mode,target_label=target_label)

if __name__ == "__main__":
    train_lightgbm()
    # main()

