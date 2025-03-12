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


## Logging
# logging set logging path
log_file_path = create_log_path('text_preprocessing')
# Create a custome logger
preprocessing_logger = CustomLogger(logger_name='text_preprocessing_logger',
                             log_filename=log_file_path)
# set logging level
preprocessing_logger.set_log_level(level = logging.INFO)



def download_nltk_resources():
    """Check and download required NLTK resources if not available."""
    resources = ["stopwords", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

def get_nlp_tools():
    """Ensure NLTK resources are available and return stopwords list and lemmatizer."""
    download_nltk_resources()
    sw = set(stopwords.words('english'))  # Using a set for faster lookups
    lemmatizer = WordNetLemmatizer()
    return sw, lemmatizer

# Nlp resources
sw, lemmatizer = get_nlp_tools()


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
    preprocessing_logger.save_logs(f"Dropping duplicates from dataframe currently shape is {df.shape} ")
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    preprocessing_logger.save_logs(f"Dropped duplicates from dataframe now shape is {df.shape} ")
    return df



def remove_punc(dataframe:pd.DataFrame) -> pd.DataFrame:
  pattern = r'[' + string.punctuation + ']'
  dataframe['text1'] = dataframe['text1'].map(lambda m:re.sub(pattern," ",m))
  dataframe['text2'] = dataframe['text2'].map(lambda m:re.sub(pattern," ",m))
  preprocessing_logger.save_logs("Removed punctuation from dataframe")
  return dataframe


def lower(dataframe:pd.DataFrame) -> pd.DataFrame:
  dataframe['text1']=dataframe['text1'].map(lambda m:m.lower())
  dataframe['text2']=dataframe['text2'].map(lambda m:m.lower())
  preprocessing_logger.save_logs("Converted text to lowercase")
  return dataframe


def tokenization(text: str) -> list:
    """Tokenizes the given text by splitting on spaces."""
    return re.split(r'\s+', text)  # Using \s+ to handle multiple spaces



def token(dataframe:pd.DataFrame) -> pd.DataFrame:
  dataframe['text1']= dataframe['text1'].apply(lambda x: tokenization(x))
  dataframe['text2']= dataframe['text2'].apply(lambda x: tokenization(x))
  preprocessing_logger.save_logs("Tokenized dataframe text columns")
  return dataframe


def remove_SW(dataframe: pd.DataFrame, sw: set) -> pd.DataFrame:
    """Removes stop words from 'text1' and 'text2' columns."""
    dataframe['text1'] = dataframe['text1'].apply(lambda x: [word for word in x if word not in sw])
    dataframe['text2'] = dataframe['text2'].apply(lambda x: [word for word in x if word not in sw])
    preprocessing_logger.save_logs("Removed stop words from dataframe")
    return dataframe


def remove_digits(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Removes numerical digits from 'text1' and 'text2' columns."""
    dataframe = dataframe.copy()  # Ensure original data remains unchanged
    dataframe['text1'] = dataframe['text1'].apply(lambda words: [word for word in words if not word.isdigit()])
    dataframe['text2'] = dataframe['text2'].apply(lambda words: [word for word in words if not word.isdigit()])
    preprocessing_logger.save_logs("Removed digits from dataframe")
    return dataframe



def read_dataframe(path:Path):
    df = pd.read_csv(path)
    return df


def save_dataframe(dataframe:pd.DataFrame,save_path:Path):
    dataframe.to_csv(save_path,index=False)
    preprocessing_logger.save_logs(f"Saved dataframe: {dataframe} at {save_path}")


# def lemmatize(dataframe:pd.DataFrame) -> pd.DataFrame:
#   dataframe['text1']=dataframe['text1'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
#   dataframe['text2']=dataframe['text2'].apply(lambda x: [lemmatizer.lemmatize(item) for item in x])
#   return dataframe

def main():
   # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # input path 
    input_path = root_path / 'data' / 'interim' 
    # save text_preprocessing path 
    save_data_path = root_path / 'data' /'processed' / 'text_preprocessed'
    # make directory 
    save_data_path.mkdir(exist_ok=True)
    # input file path
    complete_input_path = input_path / sys.argv[1]
    # read file
    df = read_dataframe(complete_input_path)
    # drop duplicate rows from data
    df_wtihout_duplicate = drop_duplicates(df)
    # remove punctutiation
    df_punc_remove = remove_punc(df_wtihout_duplicate)
    # convert text into lower case
    df_lower = lower(df_punc_remove)
    # tokenization of text columns
    df_token = token(df_lower)
    # remove stopwords 
    df_without_stopword = remove_SW(dataframe=df_token,sw=sw)
    # remove digits 
    df_without_digits = remove_digits(df_without_stopword)
    # save the transformed data
    save_dataframe(dataframe=df_without_digits,
                        save_path=save_data_path / 'train.csv')
    

if __name__=='__main__':
   main()

    










