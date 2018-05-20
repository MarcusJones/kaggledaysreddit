#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:00:01 2018

@author: batman
"""
import pandas as pd
import os
DATA_ROOT_LOCAL = "/home/batman/git/KaggleDaysReddit"
DATA_ROOT_DROPBOX = "/home/batman/git/KaggleDaysReddit"

#%% OUR SUBMISSION
path_our_submission = os.path.join(DATA_ROOT_LOCAL,'kar_submission2.csv')
our_data = pd.read_csv(path_our_submission,index_col=0)

our_data.info()
#%% LEAKED DATA
path_leaked_data = os.path.join(DATA_ROOT_LOCAL,'leaked_records.csv')
leaked_data = pd.read_csv(path_leaked_data,index_col=0)
leaked_data.info()
#%% MERGED
res=leaked_data.combine_first(our_data)
res.to_csv("kar_submission_leakagecorrected2.csv")

#df_train_sentiment = pd.read_csv("./features/train_sentiment_score.csv")
#df_test_sentiment = pd.read_csv("./features/test_sentiment_score.csv")
#train_df = pd.merge(X_train_eng, df_train_sentiment, on='id')
#test_df = pd.merge(X_test_eng, df_test_sentiment, on='id')