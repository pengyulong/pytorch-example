import pandas as pd

project_config = {
    'csvfile':"train_data_final2.csv",
    'label':"consume_purchase",
    'user_id_features':['city','city_rank','device_name','device_size','career','gender','net_type','residence','emui_dev'], #用户类特征
    'ad_id_features':['task_id','adv_id','create_type_cd','adv_prim_id','dev_id','inter_type_cd','spread_app_id','tags','app_first_class','app_second_class','indu_name'], #广告类特征
    'continue_features':['age','app_score','list_time','device_price','communication_onlinerate','communication_avgonline_30d'], #连续类特征
    'sparse_features':['city','communication_onlinerate','task_id','adv_id'], #稀疏类特征
    'dense_features':['city_rank','device_name','device_size','career','gender','net_type','residence','emui_dev','create_type_cd','adv_prim_id','dev_id','inter_type_cd','spread_app_id','tags','app_first_class','app_second_class','indu_name'] #密集类特征
}

