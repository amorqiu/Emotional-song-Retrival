#import packages
import pandas as pd
import numpy as np
import re
import random
from langdetect import detect_langs 
from label import get_label #this function can get the emotion label of an input
from rank_bm25 import BM25Okapi,BM25L, BM25Plus #this package can calculate BM25 for each document
import nltk
from nltk.tokenize import word_tokenize #this package tokenize word by whitespace and punctuation
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
#import packages for gensim
import gensim
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities

#read song data
song_df=pd.read_csv('data/emotionsong.csv').drop('Unnamed: 0',axis=1)

#fliter out no lyrics songs
song_df=song_df[song_df['lyrics']!='no lyrics']

#####Data Preprocessing
# remove round brackets and curly brackets but not text within
song_df['lyrics'] = song_df['lyrics'].map(lambda s: re.sub(r'\(|\)', '', s))
song_df['lyrics'] = song_df['lyrics'].map(lambda s: re.sub(r'\{|\}', '', s))

#remove line breaks
song_df['lyrics'] = song_df['lyrics'].map(lambda s: re.sub(r' \n|\n', '', s))

#remove non-English Lyrics
def get_eng_prob(text):
    detections = detect_langs(text)
    for detection in detections:
        if detection.lang == 'en':
            return detection.prob
    return 0

song_df['en_prob'] = song_df['lyrics'].map(get_eng_prob)
song_df = song_df.loc[song_df['en_prob'] >= 0.5]

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
song_df['lyrics'] = song_df['lyrics'].map(lower_url)

def token(lyrics):
    tokens=word_tokenize(lyrics)
    return tokens
song_df['tokens']= song_df['lyrics'].map(token)

def remove_stop(tokens):
    f = open("stoplist.txt", "r")
    stoplist=f.read()
    words = [w for w in tokens if not w in stoplist]
    return words
# remove stopwords

song_df['tokens']= song_df['tokens'].map(remove_stop)

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

song_df['tokens']= song_df['tokens'].map(lemma_token)


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
    doc_scores = get_corpus(corpus,query)
    corpus_df['rank_score']=doc_scores
    song_info=corpus_df.sort_values(by='rank_score',ascending=False)[0:10]
    return song_info

def print_songs():
    query=input("Enter your emotion: ")
    song_info=get_song_lsi(query)
    for index, row in song_info.iterrows():
        print("Song name:"+row['song'],',', "Singer Name:" + row['singer'])


print_songs()