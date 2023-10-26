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

def cleanup_df(df: pd.DataFrame,
               threshold_columns: float = 0.5,
               threshold_rows: float = 0.0,
               inplace: bool = True) -> pd.DataFrame:
    '''
    Clean up the dataframe NaN
    args:
        df: pd.DataFrame
        threshold_columns: threshold for columns nan values
        threshold_rows: threshold for rows nan values
    return:
        pd.DataFrame
    '''
    if not inplace:
        df = df.copy()
    # drop columns and rows with nan values
    df.dropna(axis=1, thresh=threshold_columns*len(df), inplace=True)
    df.dropna(axis=0, thresh=threshold_rows*len(df), inplace=True)
    return df

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

