import pandas as pd
from src.features.text_preprocessing import *
from src.features.text_normalization import *

## Logging
# Set logging path
log_file_path = create_log_path('predict_logger')
# Create a custom logger
predict_logger = CustomLogger(logger_name='predict_logger', log_filename=log_file_path)
# Set logging level
predict_logger.set_log_level(level=logging.INFO)

def preprocess_text_dataframe(data: dict) -> pd.DataFrame:
    """
    Applies all text preprocessing steps on the given DataFrame.

    :param df: Input DataFrame with 'text1' and 'text2' columns.
    :return: Processed DataFrame.
    """
    predict_logger.save_logs(msg=f'Data has come')
    df = pd.DataFrame([data])
    predict_logger.save_logs(msg=f'Data has been load')
    df = drop_duplicates(df)
    df = remove_punc(df)
    df = lower(df)
    df = token(df)
    df = remove_SW(df, sw)
    df = remove_digits(df)
    predict_logger.save_logs(msg=f'Data preprocessing done')
    return df


def normalize_text_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all text normalization steps sequentially.
    
    :param df: Input DataFrame with 'text1' and 'text2' columns.
    :return: Normalized DataFrame.
    """
    predict_logger.save_logs(msg=f'Data before lemmatize text1: {df.loc[0, 'text1'][:100]}, text2: {df.loc[0, 'text2'][:100]}')
    df = lemmatize(df)
    predict_logger.save_logs(msg=f'Data after lemmatize load text1: {df.loc[0, 'text1'][:100]}, text2: {df.loc[0, 'text2'][:100]}')
    df = remove_empty_tokens(df)
    predict_logger.save_logs(msg=f'token has been removed')
    df = remove_single_letters(df)
    predict_logger.save_logs(msg=f'single letters has been removed')
    df = detoken(df)
    predict_logger.save_logs(msg=f'detoken has been done')
    df = remove_extra_spaces(df)
    predict_logger.save_logs(msg=f'extra spaces has been removed')
    return df