import numpy as np
import pandas as pd
import time


a = np.array([5, 4, 3, 2, 1])
b = [5,1, 4, 3,3, 2, 1]
b= sorted(b, key=b.count, reverse=True)
print(b)

'''
# 验证是否会出现1,2,1这种循环的id序列
dataset = 'Cell'
df1=pd.read_csv('./my_data/Amazon_'+dataset+'/train_sessions.csv')
print(df1.head())
for i in range(len(df1['label'])):
    if df1.iloc[i,1] in [int(x) for x in df1.iloc[i,0][1:-1].split(', ')]:
        print(df1.iloc[i,1],df1.iloc[i,0],'+++++++++')
'''

'''
import time
import argparse
import pickle
from model import *
from utils import *
import pandas as pd
import os
import random
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Beauty', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=120)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)

opt = parser.parse_args()

def main():
    init_seed(2020)

    if opt.dataset == 'Beauty':
        num_node = 11674
        opt.n_iter = 1
        opt.dropout_gcn = 0.4
        opt.dropout_local = 0.5
    elif opt.dataset == 'Cell':
        num_node = 9806
        opt.n_iter = 1
        opt.dropout_gcn = 0.4
        opt.dropout_local = 0.5

    with open('my_data/Amazon_' + opt.dataset + '/' + opt.dataset + '_change.txt') as f:
        data1 = f.readlines()
    data1 = [x.split(' ', 1)[1].split('\n')[0].split(' ') for x in data1]
    seq = []
    for i in data1:
        seq.append([int(x) + 1 for x in i])  # 所有id都+1，因为在训练阶段计算loss时有一句target-1，如果id=0就会变成-1而报错
    train_data, valid_data, test_data=([],[]),([],[]),([],[])
    for sequence in seq:
        # 取倒数两个之前的列表作为训练集
        input_ids = sequence[:-1]
        for i in range(len(input_ids)-1):
            train_data[0].append(input_ids[:i+1])  # 所有子列表
            train_data[1].append(input_ids[i + 1])
        # valid_data[0].append(sequence[:-2])
        # valid_data[1].append(sequence[-2])
        test_data[0].append(sequence[:-1])
        test_data[1].append(sequence[-1])


    adj = pickle.load(open('my_data/Amazon_'+opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open('my_data/Amazon_'+opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    preds=[]
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr, preds = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]: # 按照 HR或MRR有一个是最优就存
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
            temp = preds
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
            temp = preds
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f' % (hit, mrr))
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    result = []
    for i in range(len(temp)):
        strNums = [str(x) for x in temp[i].tolist()]
        result.append('[' + ",".join(strNums) + ']')
    resultdf = pd.DataFrame(result)
    if opt.validation:
        resultdf.to_csv('my_data/Amazon_' + opt.dataset + '/gce_result.csv', index=False, header=False)
    else:
        for sequence in seq:
            # 取倒数两个之前的列表作为训练集
            input_ids = sequence
            for i in range(len(input_ids)-1):
                train_data[0].append(input_ids[:i+1])  # 所有子列表
                train_data[1].append(input_ids[i + 1])
        data_file_dir = 'my_data/Amazon_' + opt.dataset + '/test_sessions.csv'
        session = pd.read_csv(data_file_dir)
        pred_list = []
        for i in session['session']:
            # print(i,type(i))
            temp = i[1:-1]
            test_data[0].append([int(x) for x in temp.split(',')])
            test_data[1].append(-1)
        _, _, preds = train_test(model, train_data, test_data)
        temp = preds
        result = []
        for i in range(len(temp)):
            strNums = [str(x) for x in temp[i].tolist()]
            result.append('[' + ",".join(strNums) + ']')
        resultdf = pd.DataFrame(result)
        resultdf.to_csv('my_data/Amazon_' + opt.dataset + '/gce_test_result.csv', index=False, header=False)


if __name__ == '__main__':
    main()
'''