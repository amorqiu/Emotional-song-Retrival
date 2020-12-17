#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
from label import get_label


# In[55]:


df2=pd.read_csv('cleaned_query.csv').drop('Unnamed: 0',axis=1)


# In[3]:


import numpy as np


# In[203]:


# pip install langdetect


# In[61]:


from metrics import MAP, NDCG


# In[18]:


from rank_bm25 import BM25Okapi,BM25L, BM25Plus
import numpy as np
import re
import random
from langdetect import detect_langs
from nltk.tokenize import word_tokenize #this package tokenize word by whitespace and punctuation


# In[19]:


import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet


# In[20]:


song_df=pd.read_csv('data/emotionsong.csv').drop('Unnamed: 0',axis=1)


# In[21]:


q_df=pd.read_csv('data/query_rating.csv',index_col=[0]).reset_index()


# In[23]:


# song_df.head()


# In[24]:


# get testing queries and corresponding query_id
queries=q_df[['query_id','query']].drop_duplicates().set_index('query_id')


# In[25]:


queries.shape


# In[366]:


queries.index


# ### EDA

# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
ax = sns.barplot(x="rating", y="rating", data=q_df, estimator=lambda x: len(x) / len(q_df) * 100)
ax.set(ylabel="Percent")
plt.title('Rating Distribution in Annotation')


# ### Baseline

# In[27]:


def token(lyrics):
    tokens=word_tokenize(lyrics)
    return tokens


# In[28]:


# def get_songs(index,row):
#     corpus_df=song_df.copy()
#     bm25 = BM25Okapi(corpus_df)
#     doc_scores=bm25.get_scores(row['query'].split(" "))
# #     print(len(doc_scores))
# #     print(corpus_df.shape)
#     corpus_df['rank_score']=doc_scores
#     filtered=q_df[q_df['query']==row['query']]
#     song_ids=filtered['song_id'].values
#     val_score=corpus_df[corpus_df['code'].isin(song_ids)]   
#     newdf=val_score.sort_values(by='rank_score',ascending=False)[0:10]
#     origin=[filtered[filtered['song_id']==i]['rating'].values[0] for i in newdf['code']]
#     relevance=np.array([i for i in origin])
#     return relevance


# In[102]:


def get_songs(index,row,corpus):
    rating=[]
    corpus_df=song_df.copy()
    bm25 = BM25Okapi(corpus)
    doc_scores=bm25.get_scores(row['query'].split(" "))
    print(len(doc_scores))
    print(corpus_df.shape)
    corpus_df['rank_score']=doc_scores
    filtered=q_df[q_df['query']==row['query']]
    song_ids=filtered['song_id'].values
    val_score=corpus_df.sort_values(by='rank_score',ascending=False)[0:10]
    for index,row in val_score.iterrows():
        if row['code'] in song_ids:
            rating.append(filtered[filtered['song_id']==row['code']]['rating'].values[0])
        else:
            rating.append(-2)        
    relevance=rating
    return relevance


# In[93]:


queries


# In[96]:


song_df['tokens']=song_df['lyrics'].map(token)
corpus=song_df['tokens'].values


# In[111]:


precision_list=[]
ndcg_10list=[]
for index,row in queries.iterrows():
    relevance=get_songs(index,row,corpus)
    print(relevance)
    precision=MAP(relevance)
    precision_list.append(precision)
    ndcg_10=NDCG(relevance)
    ndcg_10list.append(ndcg_10)


# In[104]:


baseline['ndcg']=ndcg_10list
baseline['map']=precision_list


# In[112]:


baseline['id']=queries.index


# In[46]:


baseline=baseline.set_index('id')


# In[47]:


plot_df=baseline.merge(query, how='left',left_index=True, right_index=True)


# In[125]:


import matplotlib.pyplot as plt
# plt.plot(ndcg_10list,'g*', precision_list, 'ro')
# plt.title('Distribution of NDCG and MAP')
# plt.show()
x_list=range(20)
plt.plot(x_list, ndcg_10list, label='NDCG@10')
plt.plot(x_list, precision_list, label='MAP')
plt.legend()
plt.title('Distribution of NDCG@10 and MAP')
plt.show()


# In[108]:


base_precision=sum(precision_list)/len(precision_list)
base_ndcg=sum(ndcg_10list)/len(ndcg_10list)


# In[109]:


print(base_precision,base_ndcg)


# ### Data Preprocessing

# In[67]:


#fliter out no lyrics songs
song_df=song_df[song_df['lyrics']!='no lyrics']


# In[192]:


# this code is to check some examples of lyrics
# print(song_df['lyrics'].iloc[3])  


# #### Deal with brackets

# In[68]:


text_in_round_brackets = sum(list(song_df['lyrics'].map(lambda s: re.findall(r'\((.*?)\)',s))), [])
print('Number of round brackets: {}'.format(len(text_in_round_brackets)))


# In[ ]:


random.seed(0)
random.choices(text_in_round_brackets, k=20)


# In[185]:


text_in_square_brackets = sum(list(song_df['lyrics'].map(lambda s: re.findall(r'\[(.*?)\]',s))), [])
print('Number of square brackets: {}'.format(len(text_in_square_brackets)))


# In[193]:


text_in_curly_brackets = sum(list(song_df['lyrics'].map(lambda s: re.findall(r'\{(.*?)\}',s))), [])
print('Number of curly brackets: {}'.format(len(text_in_curly_brackets)))


# In[ ]:


random.seed(0)
random.choices(text_in_curly_brackets, k=20)


# It seems contents inside the curly bracket and round bracket are a part of the lyrics, so I will just remove the brackets and keep the contents inside

# In[196]:


# remove round brackets and curly brackets but not text within
song_df['lyrics'] = song_df['lyrics'].map(lambda s: re.sub(r'\(|\)', '', s))
song_df['lyrics'] = song_df['lyrics'].map(lambda s: re.sub(r'\{|\}', '', s))


# #### Remove Line Breaks

# In[70]:


song_df['lyrics'] = song_df['lyrics'].map(lambda s: re.sub(r' \n|\n', '', s))


# #### Remove non-English Lyrics

# In[71]:


def get_eng_prob(text):
    detections = detect_langs(text)
    for detection in detections:
        if detection.lang == 'en':
            return detection.prob
    return 0

song_df['en_prob'] = song_df['lyrics'].map(get_eng_prob)

print('Number of english songs: {}'.format(sum(song_df['en_prob'] >= 0.5)))
print('Number of non-english songs: {}'.format(sum(song_df['en_prob'] < 0.5)))


# In[72]:


song_df = song_df.loc[song_df['en_prob'] >= 0.5]


# In[73]:


def lower_url(file):
    file_lowered=file.lower() 
    file_url=re.sub(r'^https?:\/\/.*[\r\n]*', '', file_lowered, flags=re.MULTILINE) #remove url
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', file_url)
# lower vocab and remove url and emoji


# In[74]:


song_df['lyrics'] = song_df['lyrics'].map(lower_url)


# In[75]:


song_df['tokens']= song_df['lyrics'].map(token)


# In[76]:


def remove_dig(tokens):
    #remove punctuations & numbers
    filtered_words = [re.sub(r'\d+', '', word) for word in tokens]
    return filtered_words
# remove digits 


# In[77]:


import re
output = re.sub(r'\d+', '', '123hello 456world')


# In[78]:


song_df['tokens']= song_df['tokens'].map(remove_dig)


# In[79]:


def remove_stop(tokens):
    f = open("stoplist.txt", "r")
    stoplist=f.read()
    words = [w for w in tokens if not w in stoplist]
    return words
# remove stopwords


# In[80]:


song_df['tokens']= song_df['tokens'].map(remove_stop)


# In[81]:


wnl = WordNetLemmatizer()
def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
def lemma_token(tokens):
    tagged_words=nltk.pos_tag(tokens)
    new_token = []
    for i in tagged_words:
        wordnet_pos = get_pos(i[1]) #or wordnet.NOUN
        new_token.append(wnl.lemmatize(i[0],pos=wordnet_pos))
    return new_token
# lemmetize my bag of word


# In[82]:


song_df['tokens']= song_df['tokens'].map(lemma_token)


# In[10]:


import seaborn as sns
import matplotlib.pyplot as pltq
ax = sns.barplot(x="rating", y="rating", data=q_df, estimator=lambda x: len(x) / len(q_df) * 100)
ax.set(ylabel="Percent")


# In[ ]:


# ## get the code for all songs in corpus
# codes=q_df['song_id']
# codes=codes.values.tolist()

# ## using codes to get lyrics for songs to create corpus for ranking search
# corpus=[song_df[song_df['code']==i].lyrics.to_string(index=False) for i in codes]


# ### Build Models

# In[184]:


emotion=[]
for i in labels:
    if i=='0':
        emotion.append('negative')
    elif i=='1':
        emotion.append('positive')
    else:
        emotion.append('neutral')


# In[179]:


# queries.sort()
labels=['0','1','2','1','0','1','0','1','2','2','1','0','1','0','1','0','0','0','1','2']


# ### BM25

# In[87]:


def get_ranked_songs(index,row,model):
    song_copy=song_df.copy()
    label=get_label(row['query'])
    #filter out corresponding songs
    if label=='0':
        corpus_df=song_copy[song_copy['final_score']=='negative']
    elif label=='1':
        corpus_df=song_copy[song_copy['final_score']=='positive']
    else:
        corpus_df=song_copy[song_copy['final_score']=='neutral']
    corpus=corpus_df['tokens'].values
    bm25 = model(corpus)
    doc_scores=bm25.get_scores(remove_stop(row['query'].split(" ")))
    corpus_df['rank_score']=doc_scores
    filtered=q_df[q_df['query']==row['query']]
    song_ids=filtered['song_id'].values
    val_score=corpus_df[corpus_df['code'].isin(song_ids)]
    if len(val_score['code'])<10:
        corpus=song_copy['tokens'].values
        bm25=model(corpus)
        doc_scores=bm25.get_scores(remove_stop(row['query'].split(" ")))
        song_copy['rank_score']=doc_scores
        val_score=song_copy[song_copy['code'].isin(song_ids)]
    newdf=val_score.sort_values(by='rank_score',ascending=False)[0:10]
    origin=[filtered[filtered['song_id']==i]['rating'].values[0] for i in newdf['code']]
    relevance=np.array([i for i in origin])
    return relevance
        
# for index, row in queries[1:11].iterrows():


# In[86]:


queries


# In[88]:


def get_result(model):
    precision_list=[]
    ndcg_10list=[]
    for index,row in queries.iterrows():
        relevance=get_ranked_songs(index,row,model)
        precision=MAP(relevance)
        precision_list.append(precision)
        ndcg_10=NDCG(relevance)
        ndcg_10list.append(ndcg_10)
    base_precision=sum(precision_list)/len(precision_list)
    base_ndcg=sum(ndcg_10list)/len(ndcg_10list)
    return base_precision,base_ndcg,precision_list,ndcg_10list


# In[89]:


# BM25Okapi,BM25L, BM25Plus
result_bm25=get_result(BM25Okapi)


# In[ ]:


result_bm25=get_result(BM25Plus)


# In[126]:


result_bm25plus=get_result(BM25Plus)


# In[127]:


result_bm25L=get_result(BM25L)


# In[129]:


result_bm25L


# In[194]:


import matplotlib.pyplot as plt
# plt.plot(ndcg_10list,'g*', precision_list, 'ro')
# plt.title('Distribution of NDCG and MAP')
# plt.show()
x_list=range(20)
plt.plot(emotion, result_bm25L[3], '*', label='NDCG@10')
plt.plot(emotion, result_bm25L[2], 'p',label='MAP')
plt.legend()
plt.title('Distribution of NDCG@10 and MAP for BM25L ')
plt.show()


# In[90]:


result_bm25


# In[190]:


import matplotlib.pyplot as plt
# plt.plot(ndcg_10list,'g*', precision_list, 'ro')
# plt.title('Distribution of NDCG and MAP')
# plt.show()
x_list=range(20)
plt.plot(emotion, result_bm25[3],"*", label='NDCG@10')
plt.plot(emotion, result_bm25[2], "p",label='MAP')
plt.legend()
plt.title('Distribution of NDCG@10 and MAP for BM25 ')
plt.show()


# In[128]:


result_bm25plus


# In[193]:


import matplotlib.pyplot as plt
# plt.plot(ndcg_10list,'g*', precision_list, 'ro')
# plt.title('Distribution of NDCG and MAP')
# plt.show()
x_list=range(20)
plt.plot(emotion, result_bm25plus[3],"*", label='NDCG@10')
plt.plot(emotion, result_bm25plus[2], "p",label='MAP')
plt.legend()
plt.title('Distribution of NDCG@10 and MAP for BM25Plus ')
plt.show()


# ### USE LSI MODEL

# In[130]:


import gensim
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities


# In[ ]:


song_df.head()


# #### Get Corpus for All Documents

# In[158]:


def get_ranked_songl_isi(index,row):
    song_copy=song_df.copy()
    label=get_label(row['query'])
    #filter out corresponding songs
    if label=='0':
        corpus_df=song_copy[song_copy['final_score']=='negative'].reset_index(drop=True)
    elif label=='1':
        corpus_df=song_copy[song_copy['final_score']=='positive'].reset_index(drop=True)
    else:
        corpus_df=song_copy[song_copy['final_score']=='neutral'].reset_index(drop=True)
    corpus=corpus_df['tokens'].values
    doc_scores = get_corpus(corpus,row['query'])
    corpus_df['rank_score']=doc_scores
    filtered=q_df[q_df['query']==row['query']]
    song_ids=filtered['song_id'].values
    val_score=corpus_df[corpus_df['code'].isin(song_ids)]
    if len(val_score['code'])<10:
        corpus=song_copy['tokens'].values
        doc_scores=get_corpus(corpus,row['query'])
        song_copy['rank_score']=doc_scores
        val_score=song_copy[song_copy['code'].isin(song_ids)]
    newdf=val_score.sort_values(by='rank_score',ascending=False)[0:10]
    origin=[filtered[filtered['song_id']==i]['rating'].values[0] for i in newdf['code']]
    relevance=np.array([i for i in origin])
    return relevance


# In[155]:


def get_corpus(test_response,query):
#     texts_doc = [
#       [word for word in document.lower().split() if word not in stoplist]
#       for document in test_response]

# remove words that appear only once
    frequency_test = defaultdict(int)
    for text in test_response:
        for token in text:
            frequency_test[token] += 1

    texts_doc = [
        [token for token in text if frequency_test[token] > 1]
        for text in test_response
        ]
    dictionary = corpora.Dictionary(texts_doc)
    corpus = [dictionary.doc2bow(text) for text in texts_doc]
    lsi_new = models.LsiModel(corpus, id2word=dictionary, num_topics=3)
    index = similarities.MatrixSimilarity(lsi_new[corpus])  
    vec_bow = dictionary.doc2bow(query.lower().split())
    vec_lsi = lsi_new[vec_bow]  # convert the query to LSI space
    sims_1 = index[vec_lsi] 
#     newsim = sorted(enumerate(sims_1), key=lambda item: -item[1])
   
    return sims_1 # get  


# In[156]:


def get_result():
    precision_list=[]
    ndcg_10list=[]
    for index,row in queries.iterrows():
        relevance=get_ranked_songl_isi(index,row)
        precision=MAP(relevance)
        precision_list.append(precision)
        ndcg_10=NDCG(relevance)
        ndcg_10list.append(ndcg_10)
    base_precision=sum(precision_list)/len(precision_list)
    base_ndcg=sum(ndcg_10list)/len(ndcg_10list)
    return base_precision,base_ndcg,precision_list,ndcg_10list


# In[159]:


lsi_result=get_result()


# In[161]:


lsi_result


# In[185]:


import matplotlib.pyplot as plt
# plt.plot(ndcg_10list,'g*', precision_list, 'ro')
# plt.title('Distribution of NDCG and MAP')
# plt.show()
x_list=range(20)
plt.plot(emotion, lsi_result[3],"*", label='NDCG@10')
plt.plot(emotion, lsi_result[2], "p",label='MAP')
plt.legend()
plt.title('Distribution of NDCG@10 and MAP for LSI ')
plt.show()


# In[186]:


df=pd.DataFrame()


# In[187]:


df['label']=emotion
df['ndcg']=lsi_result[3]
df['map']=lsi_result[2]


# In[188]:


import matplotlib.pyplot as plt
import pandas as pd

# a scatter plot comparing num_children and num_pets
df.plot(kind='scatter',x='label',y='ndcg',color='red')
df.plot(kind='scatter',x='label',y='ndcg',color='red')
plt.show()


# In[169]:


len(labels)


# In[171]:


queries


# In[200]:


def get_song_lsi(query):
    song_copy=song_df.copy()
    label=get_label(query)
    #filter out corresponding songs
    if label=='0':
        corpus_df=song_copy[song_copy['final_score']=='negative'].reset_index(drop=True)
    elif label=='1':
        corpus_df=song_copy[song_copy['final_score']=='positive'].reset_index(drop=True)
    else:
        corpus_df=song_copy[song_copy['final_score']=='neutral'].reset_index(drop=True)
    corpus=corpus_df['tokens'].values
    doc_scores = get_corpus(corpus,row['query'])
    corpus_df['rank_score']=doc_scores
    song_info=corpus_df.sort_values(by='rank_score',ascending=False)[0:10]
    return song_info


# In[201]:


def print_songs(query):
    query=input('')
    song_info=get_song_lsi(query)
    for index, row in song_info.iterrows():
        print(row['song'],row['singer'])


# In[197]:


# song_df.head()


# In[203]:


print_songs(query)

