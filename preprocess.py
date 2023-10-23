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