# 将专家系统的结果放在最后几位
import pandas as pd

dataset='Beauty'
with open('./my_data/Amazon_'+dataset+'/'+dataset+'_change.txt','r') as f:
    data2=f.readlines()
data2=[x.split(' ',1)[1].split('\n')[0] for x in data2]
data2=[x.split(' ') for x in data2]
pred_dict={}
data3=[]
for x in data2:
    data3.append([int(i) for i in x])
# print(data3)
for x in data3:
    for i in range(len(x)-1):
        if x[i] not in pred_dict.keys():
            pred_dict[x[i]]=[]
        pred_dict[x[i]].append(x[i+1])
# print(pred_dict)
# df2=pd.DataFrame([int(x) for x in data2],columns=['label'])
# print(df2.head())

df1=pd.read_csv('./my_data/Amazon_'+dataset+'/test_result_0.csv')
# print(df1.head())
pred_list=[]
for i in df1['predict']:
    temp=i[1:-1]
    pred_list.append([int(x) for x in temp.split(',')])
print(pred_list[2])

k=-1
for i in df1['session']:
    k += 1
    x=i[1:-1].split(', ')
    x=[int(xx) for xx in x]
    if len(x)!=1:
        continue
    if x[0] not in pred_dict.keys():
        continue
    if len(pred_dict[x[0]])==1:
        pred_list[k].insert(-1,pred_dict[x[0]][0])
        pred_list[k].pop(-1)
    else: # 如果训练集里此id出现了不止一次(即此id后面的id不止一个)，则先按照后面id们的出现频数排序
        # temp=[]
        # for i in pred_dict[x[0]]:
        #     if i not in pred_list[k]:
        #         temp.append(i)
        # print(x[0],pred_dict[x[0]])
        pred_dict[x[0]] = sorted(pred_dict[x[0]], key=pred_dict[x[0]].count, reverse=True)
        temp1 = pred_list[k]
        for i in pred_dict[x[0]]:
            temp1.insert(-len(set(pred_dict[x[0]])),i)
        temp2 = list(set(temp1)) # 去重
        temp2.sort(key=temp1.index)
        pred_list[k] = temp2[:20]
        # break

print(pred_dict[105])
print(pred_list[2])
predict_result=[]
for pred in pred_list:
    # pred = eval(str(pred[0]))
    info = '['
    for pred_item in pred:
        info = info + str(pred_item) + ','
    info = info[:-1]
    info = info + ']'
    predict_result.append(info)
df3=pd.DataFrame({'session':df1['session'], 'predict':predict_result})
df3.to_csv('./my_data/Amazon_'+dataset+'/test_result.csv',index=False)