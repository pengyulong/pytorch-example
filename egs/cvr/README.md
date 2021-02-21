## CVR(转化率)预估模型实战:
### 华为广告比赛数据集描述:
- 标签:consume_purchase 付费标签(转化) 
- 广告ID类特征:
  * uid 用户id(极度稀疏)
  * task_id 广告任务id
  * adv_id 广告材料的唯一id
  * creat_type_cd 广告素材类型的唯一id
  * adv_prim_id 广告任务的广告商id
  * dev_id 广告任务的开发者id
  * inter_typ_cd 广告素材的显示形式
  * slot_id 广告位id
  * spread_app_id 广告任务的应用id
  * tags 广告任务的应用代码
  * app_first_class 广告任务的应用级别1类别
  * app_second_class 广告任务的应用级别2类别
  * indu_name 广告信息
  
- 数值类特征：
  * age 用户年龄
  * app_score 应用评分
  * list_time 型号发布时间
  * device_price 设备价格
  * communication_onlinerate 手机活动时间
  * communication_avgonline_30d 手机每日活动时间
  * pt_d 发生行为的日期

- 用户类ID特征:
  * city 用户的居住城市
  * city_rank 用户所在城市的级别
  * device_name 用户使用的手机型号
  * device_size 用户使用的手机尺寸
  * career 用户职业
  * gender 用户性别
  * net_type 发生行为时的网络状态
  * residence 用户的常住省
  * emui_dev EMUI版本

### 特征分类:
  ```Python
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
  ```

### 机器学习模型:
- 方法A:使用`lightgbm`模型单独建模:对所有非连续类特征进行one-hot编码与连续类特征进行组合输入到`lightgbm`模型中:
- 方法B:使用业务类型对特征进行分组**用户ID类特征**建立`树模型1`,**广告ID类特征**建立`树模型2`,并从这两类特征中选取重要性较高的特征与**数值类特征**组合输入到`LR`模型中进行预测
- 方法C:按照稀疏程度对特征进行分组**稀疏类特征**建立`树模型1`,**密集类特征**建立`树模型2`,并从这两类特征中选取重要性较高的特征与**数值类特征**组合输入到`LR`名中进行预测
- 比较三种建模方式的优劣,评价指标是验证集上的`logloss`,`auc`和训练时间`time`:
  | 方法 | train-logloss | train-auc | valid-logloss | valid-auc |time|
  |-----|---------------|------------|---------------|----------|-----|
  | A | 0.419|0.645|0.421|0.643|187|
  | B | 0.417|0.645|0.418|0.644|167|
  | C | 0.420|0.638|0.422|0.637|246|
### Wide&Deep 模型:
- 模型架构:
- 模型训练:
- 模型评测指标:
## DeepFM模型:
- 模型架构:
- 模型训练:
- 模型评测指标:
