import pandas as pd

dataset='Beauty'
df1=pd.read_csv('./my_data/Amazon_'+dataset+'/test_result.csv')
# print(df1.head())
df2=pd.read_csv('./my_data/Amazon_'+dataset+'/test_sessions.csv')

pred_list=[]
for i in df1['predict']:
    temp=i[1:-1]
    pred_list.append([int(x) for x in temp.split(',')])

MRR5=0;MRR20=0
HR5=0;HR20=0
total=0
print(len(pred_list),len(df2['label']))
for i in range(len(df2['label'])):
    # print(df2.iloc[i,1])
    total+=1
    if df2.iloc[i,1] in pred_list[i][:5]:
        HR5+=1
        MRR5+=1/(pred_list[i].index(df2.iloc[i,1])+1)
    if df2.iloc[i,1] in pred_list[i][:20]:
        HR20+=1
        MRR20+=1/(pred_list[i].index(df2.iloc[i,1])+1)
scores=[HR5,MRR5,HR20,MRR20]
scores=[x/total for x in scores]
a={}
a['HR5']=scores[0]
a['MRR5']=scores[1]
a['HR20']=scores[2]
a['MRR20']=scores[3]
print(a)