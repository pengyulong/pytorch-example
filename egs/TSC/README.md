# DPCNNWithAttenion网络,使用DPCNN抽取序列特征,使用Attention对多源输入进行加权平均得到聚合特征进行分类:
1. 实现了2种attention,一种是传统的`selfAttention`,一种基于Transformer的`MultiHeadAttention`
2. `attention_dpcnn_bce.py`目前综合表现最好的模型,损失函数为`BCE`,直接在安装好的环境下运行`python attention_dpcnn_bce.py`
3. `attention_dpcnn.py`支持选择`CrossEntropy`和`LabelSmooth`两种损失函数可以代码381行修改,进行切换;
4. `utils.py`是工具类函数,一般情况下不需要进行改变;
5. `visual_net.py`是对`DPCNN`特征进行可视化,在63-64选择模型参数和模型文件,然后运行`python visual_net.py`;
6. `result_28`表示通道数为28的文件夹,里面包含一个`BCE`子文件夹,跟上次发给你的结果一样,一看应该知道每个文件的意义。
