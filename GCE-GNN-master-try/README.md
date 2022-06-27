# GCE-GNN副项目

## 用处

横向切割train数据集，大体了解模型拟合情况

## 运行方法

1. 数据划分

   ~~~
   python split_traindata.py
   ~~~

2. 数据预处理

   在./GCE-GNN-master-try/my_data路径下：

   ~~~
   python data_process.py --data_name Beauty
   python data_process.py --data_name Cell
   python create_sample.py
   ~~~

   在./GCE-GNN-master-try路径下：

   ~~~
   python my_build_graph.py --dataset Beauty
   python my_build_graph.py --dataset Cell
   ~~~

3. 训练模型

   ~~~
   python my_main.py --dataset [dataset name] --validation
   ~~~

4. 模型预测

   ~~~
   python my_main.py --dataset [dataset name] --model [model name.pth]
   ~~~

5. 专家系统的融合

   ~~~
   python expertise.py
   ~~~

6. test结果验证

   ~~~
   python compare.py
   ~~~

   

