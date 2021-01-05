#!/usr/bin/env python 
#-*- coding:utf-8 _*- 
"""
@author:Chen Junhang
@license: Apache Licence
@file: config.py
@time: 2020/12/12
@contact: chenjunhang@aiyunxiao.com
"""
import sys
sys.path.append('..')
import torch

# Global
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Training
epochs = 20
batch_size = 128
eval_batch_size = 7145
context_eval_batch_size = 128
test_batch_size = 15141
# test_batch_size = 8137

learning_rate = 0.0005
lr_decay = 0.0
max_grad_norm = 2.0
max_length = 64
kernel_sizes = [1, 2, 3, 4, 5]
num_channels = [128,128,256,128,256]
hidden_size = 256
layer_num = 1
bidirectional = True
direction_num = 2 if bidirectional else 1
save_model = './save_models/'
val_losses = './save_models/' + '/val_losses.txt'
log_path = '../logs/' + 'pytorch_text_cnn.log'


if __name__ == '__main__':
    import pandas as pd
    from scipy.stats import pearsonr
    import numpy as np
    # def comment_sum(col):
    #     aa = np.mean(np.array(col['Income'])-np.array(col['Age']))
    #     return aa
    #
    # df = pd.DataFrame({'Country': ['China', 'China', 'India', 'India', 'America', 'Japan', 'China', 'India'],
    #                    'Income': [1, 1, 5, 5, 4, 5, 8, 5],
    #                    'Age': [5, 4, 1, 4, 2, 2, 4, 4]})
    # # print(df.groupby('Country').agg(comment_sum).mean()[0])
    # # print(df.groupby('Country').agg(lambda x: sum(x['Income'])-sum(x['Age'])).mean()[0])
    # print(df.groupby('Country').agg(lambda x: sum(x['Income']) - sum(x['Age'])).mean()[0])

    df = pd.DataFrame({'y': [3, 1, 3, 2, 1, 4, 2, 1,1, 1, 5, 5, 4, 5, 8, 5,3, 1],
                       'y_pred': [1, 1, 5, 5, 4, 5, 8, 5,3, 1, 3, 2, 1, 4, 2, 1,4,3],
                       'school_id': [1, 1, 1, 1, 1,1,1,1, 2, 2, 2, 2,2,2,2,2,2,2],
                       'exam_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3,3,4, 4,4,4]})

    def pp(x):
        ppc, _ = pearsonr(x['y'], x['y_pred'])
        return ppc
    # print(df.groupby(['school_id','exam_id']).agg(pp).mean()[0])
    gg = df.groupby(['school_id','exam_id'])
    dd = []
    for name, group in gg:
        print(name)
        print(group)
        print(pearsonr(group['y'], group['y_pred'])[0])
        dd.append(pearsonr(group['y'], group['y_pred'])[0])
    print(np.nanmean(dd))
    pcc = np.nanmean([pearsonr(x['y'], x['y_pred'])[0] for _,x in gg])
    print(pcc)
    ppc_map = np.nanmean(list(map(lambda x: pearsonr(x[1]['y'], x[1]['y_pred'])[0], gg)))
    print(ppc_map)
