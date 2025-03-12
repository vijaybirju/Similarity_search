import sys
import re
import ast
import nltk
import string
import joblib
import logging
import numpy as np
import pandas as pd 
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from src.logger import create_log_path, CustomLogger

## Logging
# Set logging path
log_file_path = create_log_path('text_normalization')
# Create a custom logger
normalize_logger = CustomLogger(logger_name='text_normalization_logger', log_filename=log_file_path)
# Set logging level
normalize_logger.set_log_level(level=logging.INFO)

lemmatizer = WordNetLemmatizer()


def log_first_row(df: pd.DataFrame, message: str):
    """Logs the first row of the dataframe with a message."""
    if not df.empty:
        normalize_logger.save_logs(f"{message} - First row before:\n{df.iloc[0].to_dict()}")

def convert_to_list(text):
    """Converts a string representation of a list into an actual list."""
    if isinstance(text, str):
        try:
            return ast.literal_eval(text)  # Converts stringified list to an actual list
        except (SyntaxError, ValueError):
            return []  # Return an empty list if conversion fails
    return text  # Return as-is if already a list


def lemmatize(df: pd.DataFrame) -> pd.DataFrame:
    """Lemmatizes words in 'text1' and 'text2' columns."""
    log_first_row(df, "Lemmatization")

    # Ensure the columns are lists
    df['text1'] = df['text1'].apply(convert_to_list)
    df['text2'] = df['text2'].apply(convert_to_list)

    df['text1'] = df['text1'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
    df['text2'] = df['text2'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])
    normalize_logger.save_logs(f"Lemmatized text - First row after:\n{df.iloc[0].to_dict()}")
    return df


def remove_empty_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Removes empty tokens from 'text1' and 'text2'."""
    log_first_row(df, "Removing empty tokens")
    df['text1'] = df['text1'].apply(lambda tokens: [word for word in tokens if word.strip()])
    df['text2'] = df['text2'].apply(lambda tokens: [word for word in tokens if word.strip()])
    normalize_logger.save_logs(f"Removed empty tokens - First row after:\n{df.iloc[0].to_dict()}")
    return df


def remove_single_letters(df: pd.DataFrame) -> pd.DataFrame:
    """Removes single-character tokens from 'text1' and 'text2'."""
    log_first_row(df, "Removing single letters")
    df['text1'] = df['text1'].apply(lambda tokens: [word for word in tokens if len(word) > 1])
    df['text2'] = df['text2'].apply(lambda tokens: [word for word in tokens if len(word) > 1])
    normalize_logger.save_logs(f"Removed single letters - First row after:\n{df.iloc[0].to_dict()}")
    return df


def detoken(df: pd.DataFrame) -> pd.DataFrame:
    """Detokenizes text in 'text1' and 'text2'."""
    detokenizer = TreebankWordDetokenizer()
    log_first_row(df, "Detokenizing")
    df['text1'] = df['text1'].apply(detokenizer.detokenize)
    df['text2'] = df['text2'].apply(detokenizer.detokenize)
    normalize_logger.save_logs(f"Detokenized text - First row after:\n{df.iloc[0].to_dict()}")
    return df


def remove_extra_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """Removes extra spaces from text."""
    log_first_row(df, "Removing extra spaces")
    df['text1'] = df['text1'].apply(lambda text: text.replace("  ", " "))
    df['text2'] = df['text2'].apply(lambda text: text.replace("  ", " "))
    normalize_logger.save_logs(f"Removed extra spaces - First row after:\n{df.iloc[0].to_dict()}")
    return df


def read_dataframe(path: Path):
    df = pd.read_csv(path)
    return df


def save_dataframe(df: pd.DataFrame, save_path: Path):
    df.to_csv(save_path, index=False)
    normalize_logger.save_logs(f"Saved dataframe at {save_path}")

def main():
    # current file path

    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    # input path 
    input_path = root_path / 'data' / 'processed' / 'text_preprocessed'
    # save text_preprocessing path 
    save_data_path = root_path / 'data' /'processed' / 'final'
    # make directory 
    save_data_path.mkdir(exist_ok=True)
    # input file path
    complete_input_path = input_path / sys.argv[1]
    # read the data
    df = read_dataframe(complete_input_path)
    # lemmatize the data
    df_lemmatized = lemmatize(df)
    # remove empty token 
    df_empty_token = remove_empty_tokens(df_lemmatized)
    # remove single letters
    df_to_detoken = remove_single_letters(df_empty_token)
    # detoken
    df_detoken = detoken(df_to_detoken)
    # remove extra space
    df_final = remove_extra_spaces(df_detoken)
    # save the transformed data
    save_dataframe(df=df_final,
                        save_path=save_data_path / 'train.csv')
    
if __name__ == '__main__':
    main()









