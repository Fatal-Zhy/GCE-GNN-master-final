import pandas as pd

dataset='Beauty'
df1=pd.read_csv('./my_data/Amazon_'+dataset+'/gce_result.csv',header=None)
# print(df1.head())

pred_list=[]
for i in df1[0]:
    temp=i[1:-1]
    pred_list.append([int(x) for x in temp.split(',')])

for i in pred_list:
    if -1 in i:
        print(i)

# df2=pd.read_csv('./data/Amazon_Beauty/train_sessions.csv')
with open('./my_data/Amazon_'+dataset+'/'+dataset+'_change.txt','r') as f:
    data2=f.readlines()
data2=[x.split(' ',1)[1].split('\n')[0] for x in data2]
data2=[x.split(' ')[-1] for x in data2]
df2=pd.DataFrame([int(x) for x in data2],columns=['label'])
# print(df2)
# print(df2.head())
MRR5=0;MRR20=0
HR5=0;HR20=0
total=0
print(len(pred_list),len(df2['label']))
for i in range(len(df2['label'])):
    # print(df2.iloc[i,1])
    total+=1
    if df2.iloc[i,0] in pred_list[i][:5]:
        HR5+=1
        MRR5+=1/(pred_list[i].index(df2.iloc[i,0])+1)
    if df2.iloc[i,0] in pred_list[i][:20]:
        HR20+=1
        MRR20+=1/(pred_list[i].index(df2.iloc[i,0])+1)
scores=[HR5,MRR5,HR20,MRR20]
scores=[x/total for x in scores]
a={}
a['HR5']=scores[0]
a['MRR5']=scores[1]
a['HR20']=scores[2]
a['MRR20']=scores[3]
print(a)