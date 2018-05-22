# =============================================================================
# Standard imports
# =============================================================================
import tarfile
import urllib as urllib
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('Started logging')

# =============================================================================
# External imports - reimported for code completion! 
# =============================================================================
print_imports()
import pandas as pd # Import again for code completion
import numpy as np # Import again for code completion
import matplotlib as mpl
import matplotlib.pyplot as plt
#sns.set(style="ticks")
import sklearn as sk
import sklearn
import sklearn.linear_model
# to make this notebook's output stable across runs
np.random.seed(42)
from sklearn_pandas import DataFrameMapper
from sklearn_features.transformers import DataFrameSelector

# Ignore useless warnings (see SciPy issue #5998)
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# Plotly
import plotly.plotly as py
py.plotly.tools.set_credentials_file(username='notbatman', api_key='1hy2cho61mYO4ly5R9Za')
import plotly.graph_objs as go

#%% 
# Rename for Karthick
train_df = X_train
test_df = X_test
#del X_train,X_test


#%%********************************************
# Transformers!
#**********************************************

#%% 
class WordCounter(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    """
    def __init__(self, col_name, new_col_name):
        self.col_name = col_name
        self.new_col_name = new_col_name
    def fit(self, X, y=None):
        return self
    
    def transform(self, df, y=None):

        df[self.new_col_name] = df[self.col_name].apply(lambda x: len(x.split(" ")) )
        #raise
        return df

# Debug:
#df = X_train
#col_name = 'question_text'
#new_col_name = "no_of_words_in_question"
#word_counter = WordCounter(col_name,new_col_name)
#word_counter.transform(df)

#%% 
class TimeProperty(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    """
    def __init__(self, time_col_name, new_col_name,time_property):
        self.time_col_name = time_col_name
        self.new_col_name = new_col_name
        self.time_property = time_property
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, df, y=None):
        original_shape=df.shape
        if self.time_property == 'hour':
            df[self.new_col_name] = df[self.time_col_name].dt.hour
        elif self.time_property == 'month':
            df[self.new_col_name] = df[self.time_col_name].dt.month
        elif self.time_property == 'dayofweek':
            df[self.new_col_name] = df[self.time_col_name].dayofweek
        else:
            raise
        print("Transformer:", type(self).__name__, original_shape, "->", df.shape)
        print("\t",vars(self))
        return df

    
# Debug:
#df = X_train
#time_col_name = 'question_utc'
#new_col_name = 'question_hour'
#time_property = 'hour'
#time_col_name = 'question_utc'
#new_col_name = 'question_month'
#time_property = 'month'
#time_adder = TimeProperty(time_col_name,new_col_name,time_property)
#res=time_adder.transform(df)
#        
#%% 
class AnswerDelay(sk.base.BaseEstimator, sk.base.TransformerMixin):
    """
    """
    def __init__(self, new_col_name,divisor=1):
        self.new_col_name = new_col_name
        self.divisor = divisor
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, df, y=None):
        df[self.new_col_name] = train_df['answer_utc']- train_df['question_utc']
        df[self.new_col_name] = df[self.new_col_name].dt.seconds/self.divisor
        return df
           
    
# Debug:
#df = X_train
#new_col_name = 'answer_delay_seconds'
#answer_delay_adder = AnswerDelay(new_col_name)
#res=answer_delay_adder.transform(df)
#        
    
#%%********************************************
# Pipeline!
#**********************************************

feature_adding_pipeline = sk.pipeline.Pipeline([
        ('answer counts', ValueCounter('question_id')),
        ('subreddit counts', ValueCounter('subreddit')),
        ('count words in question', WordCounter('question_text','no_of_words_in_question')),
        ('count words in answer', WordCounter('answer_text','no_of_words_in_answer')),
        ('question hour', TimeProperty('question_utc','question_hour','hour')),
        ('question day', TimeProperty('question_utc','question_day','hour')),
        ('question month', TimeProperty('question_utc','question_month','hour')),
        ('Answer delay seconds', AnswerDelay('answer_delay_seconds',divisor=1)),
        ])

for i,step in enumerate(feature_adding_pipeline.steps):
    print(i,step)

#%% Apply the pipeline
X_train_eng = feature_adding_pipeline.fit_transform(X_train)
X_test_eng = feature_adding_pipeline.fit_transform(X_test)

#%% Add the sentiment
df_train_sentiment = pd.read_csv("./features/train_sentiment_score.csv")
df_test_sentiment = pd.read_csv("./features/test_sentiment_score.csv")



train_df = pd.merge(X_train_eng, df_train_sentiment, on='id')
test_df = pd.merge(X_test_eng, df_test_sentiment, on='id')

#%% Save to CSV
train_df.to_csv('dataset_train_simple_and_sentiment.csv')
test_df.to_csv('dataset_test_simple_and_sentiment.csv')

#%% DONE HERE 

print("******************************")
raise
