#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 13:38:40 2018

@author: batman
"""

import re
import pandas as pd
import numpy as np
import os
import sklearn.model_selection
import sklearn.feature_extraction#.text# import CountVectorizer

#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import (
#    CountVectorizer, TfidfVectorizer, HashingVectorizer
#)

#import warnings
#warnings.filterwarnings("ignore")

#%% DATA

DATA_ROOT = r"~/Dropbox/DATA/Pawal Model Pipeline Data/data"
df = pd.read_csv(os.path.join(DATA_ROOT,"ebaytitles.csv"))
df_desc = df.describe()

df_counts = df.groupby('category_name').count().sort_values('title',ascending=False)
df_counts.columns = ['count']
df_counts['freq percent'] = df_counts['count']/df_counts['count'].sum() * 100

# OR:
#df_counts = df['category_name'].value_counts()#

#%% SUBSET
df = df.sample(frac=0.1)

df_counts = df.groupby('category_name').count().sort_values('title',ascending=False)
df_counts.columns = ['count']
df_counts['freq percent'] = df_counts['count']/df_counts['count'].sum() * 100


#%% Train Test split
#X = df.title.values
#y = df.category_name.values
#
## Make the split
#X_tr, X_te, y_tr, y_te = sk.model_selection.train_test_split(X, 
#                                          y,
#                                          test_size=0.1,
#                                          random_state=0)
#
## Recreate the DFs
#Xy_tr = pd.DataFrame({'title':X_tr,'category_name':y_tr})
#Xy_te = pd.DataFrame({'title':X_te,'category_name':y_te})
#
## Get the counts
#df_counts_train = Xy_tr.groupby('category_name').count().sort_values('title',ascending=False)
#df_counts_train.columns = ['count train']
#df_counts_train['freq percent train'] = df_counts_train['count train']/df_counts_train['count train'].sum() * 100
#df_counts_test = Xy_te.groupby('category_name').count().sort_values('title',ascending=False)
#df_counts_test.columns = ['count test']
#df_counts_test['freq percent test'] = df_counts_test['count test']/df_counts_test['count test'].sum() * 100
#
#df_counts1 = pd.concat([df_counts,df_counts_train,df_counts_test],axis=1)
#
#del X,y, X_tr, X_te, y_tr, y_te,df_counts_train,df_counts_test

#%% STRATIFIED Train Test split
X = df.title.values
y = df.category_name.values

X_tr, X_te, y_tr, y_te = sk.model_selection.train_test_split(X, 
                                          y,
                                          test_size=0.1,
                                          random_state=0,stratify=y)

# Recreate the DFs
Xy_tr = pd.DataFrame({'title':X_tr,'category_name':y_tr})
Xy_te = pd.DataFrame({'title':X_te,'category_name':y_te})

# Get the counts
df_counts_train = Xy_tr.groupby('category_name').count().sort_values('title',ascending=False)
df_counts_train.columns = ['count train']
df_counts_train['freq percent train'] = df_counts_train['count train']/df_counts_train['count train'].sum() * 100
df_counts_test = Xy_te.groupby('category_name').count().sort_values('title',ascending=False)
df_counts_test.columns = ['count test']
df_counts_test['freq percent test'] = df_counts_test['count test']/df_counts_test['count test'].sum() * 100

df_counts1 = pd.concat([df_counts,df_counts_train,df_counts_test],axis=1)

del X,y, X_tr, X_te, y_tr, y_te,df_counts_train,df_counts_test

#%% Count vectorizer
count_vect = sk.feature_extraction.text.CountVectorizer()
#small_X_tr = 
X_train_counts = count_vect.fit_transform(Xy_tr.loc[:,'title'])#.todense()
X_train_counts_shape = X_train_counts.shape

# Get vocab and create a DF
vocab = count_vect.vocabulary_
vocab_sorted = sorted([(vocab[k], k) for k in vocab], key=lambda t:t[1])
vocab_sorted = list([pair[1] for pair in vocab_sorted])

# A sample of these results
df_vocab_sample_counts = pd.DataFrame(data=X_train_counts[0:10,0:10000].todense(),columns=vocab_sorted[0:10000])

# Some words are very infrequent
total_words = np.sum(X_train_counts)
X_train_counts_summed = np.sum(X_train_counts,axis=0)
X_train_counts_summed = np.squeeze(np.asarray(X_train_counts_summed))
word_counts = pd.Series(X_train_counts_summed).value_counts()

# Frequencies of all words
np.sum(X_train_counts_summed == 1)
np.sum(X_train_counts_summed == 2)
X_train_counts_fracs = X_train_counts_summed/ total_words
#pd.
df_total_vocab_freqs=pd.DataFrame(data=X_train_counts_fracs.T,
                                  index=vocab_sorted,columns=['freq']).sort_values('freq',ascending=False)

#%% Frequency vectorizer
freq_vect = sk.feature_extraction.text.TfidfVectorizer()
#small_X_tr = 
X_train_freqs = freq_vect.fit_transform(Xy_tr.loc[:,'title'])#.todense()
X_train_freqs_shape = X_train_freqs.shape
df_vocab_sample_freqs = pd.DataFrame(data=X_train_freqs[0:10,0:10000].todense(),columns=vocab_sorted[0:10000])

#freqs_total = np.sum(X_train_freqs,axis=0)



