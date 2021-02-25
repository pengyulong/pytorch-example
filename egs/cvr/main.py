import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score,f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM,WDL

import numpy as np


def evaluate_model(y_true,y_pred):
    best_f1, best_thresh = 0.0, 0.0
    for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        y_pred = np.where(y_pred>thresh,1,0)
        prec, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary")
        tn, fp, fn, tp = confusion_matrix(y_true,y_pred).ravel()
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
        print("model output:precision:{},recall:{},f1:{},thresh:{}".format(
            prec, recall, f1, thresh))
        print("tn:{},fp:{},fn:{},tp:{}".format(tn,fp,fn,tp))
    print("best thresh:{},best f1:{}".format(best_thresh, best_f1))
    return best_thresh


if __name__ == "__main__":
    # data_type = {'in_car_x':np.int16,'in_car_y':np.int16,'gender_body_x':np.int16,'gender_body_y':np.int16,'upper_wear_body_x':np.int16,'upper_wear_body_y':np.int16,
    #              'upper_color_body_x':np.int16,'upper_color_body_y':np.int16,'headwear_body_x':np.int16,'headwear_body_y':np.int16,'face_mask_body_x':np.int16,
    #              'face_mask_body_y':np.int16,'glasses_body_x':np.int16,'glasses_body_y':np.int16,'same_date':np.int16,'sim':np.float64,'sim2':np.float64,'blur_x':np.float64,
    #              'blur_y':np.float64,'distance':np.float64,'cam_name_dist':np.float64,'car_dist':np.float64,'facescore_x':np.float64,'facescore_y':np.float64}

    train = pd.read_csv(r'./data/raw/train.csv')
    test = pd.read_csv(r'./data/raw/test.csv')
    # print(data.head())

    sparse_features = ['in_car_x','in_car_y','gender_body_x','gender_body_y','upper_wear_body_x','upper_wear_body_y','upper_color_b