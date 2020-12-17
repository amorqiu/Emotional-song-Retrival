# import packages
import pandas as pd
import numpy as np
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# import pre-generated csv
df2=pd.read_csv('cleaned_query.csv')

# generate tokenizer
max_fatures = 500000
tokenizer = Tokenizer(num_words=max_fatures, split=' ', lower=True)
tokenizer.fit_on_texts(df2['cleaned_sentence'].values)

# import pre-trained model
model = tf.contrib.keras.models.load_model( 'Mymodel.h5' )
def get_label(x):
    seq = tokenizer.texts_to_sequences([x])
    padded = pad_sequences(seq, maxlen=50, dtype='int32', value=0)
    pred = model.predict(padded)
    labels = ['0','1','2']
    return labels[np.argmax(pred)]