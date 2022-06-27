import pickle
import argparse


dataset = 'Beauty'
sample_num = 12

with open('my_data/Amazon_'+dataset+'/'+dataset+'_change.txt') as f:
    data1=f.readlines()
data1=[x.split(' ',1)[1].split('\n')[0].split(' ') for x in data1]
seq=[]
for i in data1:
    seq.append([int(x)+1 for x in i]) #所有id都+1，因为在训练阶段计算loss时有一句target-1，如果id=0就会变成-1而报错

# with open('my_seq.txt','w') as f:
#     f.write(str(seq))
# for i in seq:
#     if 0 in i:
#         print(i)

temp=set()
for i in seq:
    temp=temp|set(i)
# k=-1 # item_id不是连号的
# for i in temp:
#     if i!=k+1:
#         print(i)
#     k=i
# num=len(temp)+1
num=max(temp)+1
print(len(temp),max(temp)) #Beauty: 11622 11672       Cell:9783 9804  | 所有id+1后:Beauty: 11622 11673

relation = []
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

weight = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

pickle.dump(adj, open('my_data/Amazon_'+dataset+ '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('my_data/Amazon_'+dataset+ '/num_' + str(sample_num) + '.pkl', 'wb'))
