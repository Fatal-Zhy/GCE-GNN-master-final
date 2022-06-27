# GCE-GNN在短序列推荐算法中的应用

## 问题描述

数据集来自于 Amazon 电商平台，包括 Beauty 和 Cellphones 两个 类别。两个数据集均已划分好训练数据和测试数据，数据中每一行为一条训练样 本。其中 train_sessions.csv 包括两个字段，session 字段为用户的历史匿名交互序 列，label 为真实数据中用户下一次交互的产品；test_sessions.csv 中仅包括 session 字段。

原始数据位于./GCE-GNN-master-final/my_data/Amazon_Beauty和./GCE-GNN-master-final/my_data/Amazon_Cell文件夹下

## 代码

本代码基于SIGIR 2020 Paper: _Global Context Enhanced Graph Neural Networks for Session-based Recommendation_.以及[GitHub源码](https://github.com/CCIIPLab/GCE-GNN)，根据本应用问题特性改编而来。

~~~~
@inproceedings{wang2020global,
    title={Global Context Enhanced Graph Neural Networks for Session-based Recommendation},
    author={Wang, Ziyang and Wei, Wei and Cong, Gao and Li, Xiao-Li and Mao, Xian-Ling and Qiu, Minghui},
    booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages={169--178},
    year={2020}
}
~~~~

## 项目结构说明

> GCE-GNN-master-final
>
> > GCE-GNN-master-try：独立于项目主体的副项目，只使用了原数据集的train_session.csv构建训练集、验证集、测试集
> >
> > my_data
> >
> > > Amazon_Beauty：存放有关Beauty数据集的所有信息（原始数据、图数据、模型、预测结果）
> > >
> > > Amazon_Cell：存放有关Cell数据集的所有信息（原始数据、图数据、模型、预测结果）
> > >
> > 
> > 各种.py文件：GCE-GNN主体

## 运行方法

1. 数据预处理

   在./GCE-GNN-master-final/my_data路径下：

   ~~~
   python data_process.py --data_name Beauty
   python data_process.py --data_name Cell
   python create_sample.py
   ~~~

   在./GCE-GNN-master-final路径下：

   ~~~
   python my_build_graph.py --dataset Beauty
   python my_build_graph.py --dataset Cell
   ~~~

2. 训练模型

   ~~~
   python my_main.py --dataset [dataset name] --validation
   ~~~

   valid结果验证

   ~~~
   python compare.py
   ~~~

3. 模型预测

   ~~~
   python my_main.py --dataset [dataset name] --model [model_name.pth]
   ~~~

4. 专家系统的融合

   ~~~
   python expertise.py
   ~~~





