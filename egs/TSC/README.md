# 多源输入时序分类任务(TSC)
这是一个多源输入的时间序列分类任务,数据来源于制作工艺品的回转窑的传感器数据,来预测最终制作的工艺品否正常,
其中传感器数据包括窑头窑尾温度,风量,喂煤量等7个传感器数据,工艺品标签包括两类异常和一类正常,比例接近3:3:14。
1. 本项目采用DPCNN抽取多源输入的时序特征,然后采用attention实现动态加权聚合,最后使用softmax对其进行分类,基本网络架构图如下:
   (T1,T2,T3,T4,T5,T6,T7)->DPCNN->(f1,f2,f3,f4,f5,f6,f7)->concat->attention->H->softmax(3)
2. 实现了2种attention,一种是传统的`selfAttention`,一种基于Transformer的`MultiHeadSelfAttention`;
3. 尝试了3种损失函数,一种是普通的`CrossEntropy`,一种是基于`soft`标签的`LabelSmooth`,还有一种是解决分类不均衡的`BinaryCrossEntropyWithLogist`,其中`BCE`表现最好,`f1`可达0.873比其他高出约10个百分点;
4. `attention_dpcnn_bce.py`实现的事损失函数为`BCE`的`attention_dpcnn`网络模型;
5. `attention_dpcnn.py`支持选择`CrossEntropy`和`LabelSmooth`两种损失函数可以代码381行修改,进行切换;
6. `utils.py`是工具类函数,一般情况下不需要进行改变;
7. `visual_net.py`是对`DPCNN`特征进行可视化;
