import sys
import re
import nltk
import string
import joblib
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
# from yaml import safe_load
import pandas as pd
from pathlib import Path
from src.logger import create_log_path, CustomLogger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize.treebank import TreebankWordDetokenizer


## Logging
# logging set logging path
log_file_path = create_log_path('data_preprocessing')
# Create a custome logger
preprocessing_logger = CustomLogger(logger_name='preprocessing_logger',
                             log_filename=log_file_path)
# set logging level
preprocessing_logger.set_log_level(level = logging.INFO)


def save_transformer(path,object):
    joblib.dump(value=object,
                filename=path)
    

def drop_duplicates(dataframe: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a dataframe.
    
    :param dataframe: Input DataFrame
    :param subset: List of column names to consider for identifying duplicates. If None, all columns are used.
    :return: DataFrame with duplicates removed
    """
    df = dataframe.copy()
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    
    return df

nltk.download('stopwords')
nltk.download('wordnet')
sw=nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def remove_punc(dataframe:pd.DataFrame) -> pd.DataFrame:
  pattern = r'[' + string.punctuation + ']'
  dataframe['text1'] = dataframe['text1'].map(lambda m:re.sub(pattern," ",m))
  dataframe['text2'] = dataframe['text2'].map(lambda m:re.sub(pattern," ",m))
  return dataframe


def lower(dataframe:pd.DataFrame) -> pd.DataFrame:
  dataframe['text1']=dataframe['text1'].map(lambda m:m.lower())
  dataframe['text2']=dataframe['text2'].map(lambda m:m.lower())
  return dataframe


def tokenization(text: str) -> list:
    """Tokenizes the given text by splitting on spaces."""
    return re.split(r'\s+', text)  # Using \s+ to handle multiple spaces



def token(dataframe:pd.DataFrame) -> pd.DataFrame:
  dataframe['text1']= dataframe['text1'].apply(lambda x: tokenization(x))
  dataframe['text2']= dataframe['text2'].apply(lambda x: tokenization(x))
  return dataframe


def remove_SW(dataframe: pd.DataFrame, sw: set) -> pd.DataFrame:
    """Removes stop words from 'text1' and 'text2' columns."""
    dataframe['text1'] = dataframe['text1'].apply(lambda x: [word for word in x if word not in sw])
    dataframe['text2'] = dataframe['text2'].apply(lambda x: [word for word in x if word not in sw])
    return dataframe


def remove_digits(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Removes numerical digits from 'text1' and 'text2' columns."""
    dataframe = dataframe.copy()  # Ensure original data remains unchanged
    dataframe['text1'] = dataframe['text1'].apply(lambda words: [word for word in words if not word.isdigit()])
    dataframe['text2'] = dataframe['text2'].apply(lambda words: [word for word in words if not word.isdigit()])
    return dataframe


def lemmatize(dataframe:pd.DataFrame) -> pd.DataFrame:
  dataframe['text1']=dataframe['text1'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
  dataframe['text2']=dataframe['text2'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
  return dataframe







