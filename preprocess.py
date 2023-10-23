import pandas as pd
from pathlib import Path

# list of exportable functions
__all__ = ['read_data', 'cleanup_df']

def read_data(path: Path) -> pd.DataFrame:
    '''
    Read data from path
    args:
        path: Path to csv
    return:
        pd.DataFrame
    '''
    df = pd.read_csv(path)
    print(df.head())
    return df

def cleanup_df(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean up the dataframe NaN
    args:
        df: pd.DataFrame
    return:
        pd.DataFrame
    '''
    pass

def encode_categorical(df: pd.DataFrame, column_names: list) -> pd.DataFrame:
    '''
    Encode categorical data
    args:
        df: pd.DataFrame
        column_names: list of column names
    return:
        pd.DataFrame
    '''
    pass

def encode_numerical(df: pd.DataFrame, column_names: list) -> pd.DataFrame:
    '''
    Encode numerical data
    args:
        df: pd.DataFrame
        column_names: list of column names
    return:
        pd.Dataframe
    '''
    pass

def encode_target(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    '''
    Encode target data
    args:
        df: pd.DataFrame
        column_name: column name of target
    return:
        pd.DataFrame
    '''
    pass

