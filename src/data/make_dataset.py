import sys
import sys
import logging
# from yaml import safe_load
import pandas as pd
from pathlib import Path
from src.logger import create_log_path, CustomLogger


log_file_path = create_log_path('make_dataset')
# create custom logger object

dataset_logger = CustomLogger(logger_name='make_dataset',
                              log_filename=log_file_path)



# set the logging level info
dataset_logger.set_log_level(level=logging.INFO)


def load_raw_data(input_path: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(input_path)
    rows, columns = raw_data.shape
    dataset_logger.save_logs(msg=f'{input_path.stem} data read having { rows} rows and {columns} columns'
                             , log_level='info')
    return raw_data


def save_data(data: pd.DataFrame,
              output_path: Path):
    data.to_csv(output_path, index=False)
    dataset_logger.save_logs(msg=f"{output_path.stem + output_path.suffix} data saved successfully to the output folder",
                             log_level='info')



def main():
    # read the input file name  from the command 
    input_file_name = sys.argv[1]
    print(f"Looking for root at: {input_file_name}")
    # current file path
    current_path = Path(__file__)
    # root directory path
    root_path = current_path.parent.parent.parent
    print(f"Looking for root at: {root_path.resolve()}")
    # interim directory path
    interim_data_path = root_path / 'data' / 'interim'
    # make directory for the interim path
    interim_data_path.mkdir(exist_ok=True)
    # row train file path
    raw_df_path = root_path / 'data' / 'raw' / input_file_name
    # load the training file 
    raw_df = load_raw_data(input_path= raw_df_path)
    # save the train data to the output path
    save_data(data= raw_df, output_path= interim_data_path / 'train.csv')

    
    
if __name__ == '__main__':
    main()
