# “阿里灵杰”问天引擎电商搜索算法赛 SimCSE版本BaseLine
使用SimCSE有监督版本的问天搜索算法赛BaseLine，无困难负样本。

比赛地址：https://tianchi.aliyun.com/competition/entrance/531946/introduction
## 文件构成
1.data_process.py --- 数据预处理(因为给出的数据列数发生过改变，这里可能要注意一下编号问题 需要对应数据修改一下)

2.train.py --- 训练主程序	

3.get_embedding.py --- 生成对应的embedding文件

4.tar.sh 压缩脚本

run_sup_example.sh 训练脚本

./models.py 模型文件

./trainers.py 训练主程序

./data/* 训练数据

## 执行顺序

1.data_precess.py ---> run_sup_example.sh ---> 3.get_embedding.py ---> 4.tar.sh



## References
<a id="1">[1]</a> 
SimCSE: Simple Contrastive Learning of Sentence Embeddings
Tianyu Gao*, Xingcheng Yao*, Danqi Chen.
In EMNLP 2021

<a id="1">[2]</a> 
https://github.com/princeton-nlp/SimCSE
