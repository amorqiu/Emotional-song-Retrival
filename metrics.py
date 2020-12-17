

import pandas as pd
import numpy as np


def NDCG(true_relevance):
    new_df=pd.DataFrame()
    rank=1
    for i in true_relevance:
        new_df=new_df.append(pd.Series([rank,i+2]),ignore_index=True)
        rank +=1
    new_df=new_df.rename(columns={0:'rank',1:'rating'})
    new_df['DCG']=2**new_df['rating']/np.log2(new_df['rank']+1)
    sorted_df=new_df.sort_values(by=['rating'],ascending=False).reset_index(drop='true')
    # print(sorted_df)
    for index,row in sorted_df.iterrows():
        # print(row['rating'],np.log2(index+2))
        row['DCG']=2**row['rating']/np.log2(index+2)
    NDCG=new_df['DCG'].sum()/sorted_df['DCG'].sum()
    return NDCG

def MAP(relevance):
    num=1
    total=1
    precision=0
    for i in relevance:
        if i>1:
            precision=precision+total/num
#             print(precision)
            total+=1
        num+=1
#         print(num)
    return precision/total
