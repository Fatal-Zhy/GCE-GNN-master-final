import numpy as np
import pandas as pd
import random

np.random.seed(2022)

split_rate=0.05
for dataset in ['Beauty','Cell']:
    df=pd.read_csv('./my_data_origin/Amazon_'+dataset+'/train_sessions.csv')
    # print(df.info())
    df_test=df.sample(frac=split_rate,replace=False)
    df_train=df[~df.index.isin(df_test.index)]
    # print(df_test.info())
    # print(df_train.info())
    df_train.to_csv('./my_data/Amazon_'+dataset+'/train_sessions.csv',index=None)
    df_test.to_csv('./my_data/Amazon_'+dataset+'/test_sessions.csv',index=None)
