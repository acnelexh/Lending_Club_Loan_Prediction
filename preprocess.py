import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import preprocessing

# list of exportable functions
__all__ = ['read_data', 'cleanup_df', 'encode_numerical', 'encode_categorical_cardinal', 'merge_labels']

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

def encode_categorical(df: pd.DataFrame, column_names: list, handle_nan: str = 'mode') -> pd.DataFrame:
    '''
    Encode categorical data
    args:
        df: pd.DataFrame
        column_names: list of column names
        handle_nan: how to handle nan values
    return:
        pd.DataFrame
    '''
    onehot_encoder = preprocessing.OneHotEncoder()
    categorical_df = df[column_names]
    # replace nan with mode for each column
    if handle_nan == 'mode':
        categorical_df.fillna(categorical_df.mode(), inplace=True)
    # encode categorical data
    categorical_df = onehot_encoder.fit_transform(categorical_df)
    return categorical_df, onehot_encoder.get_feature_names_out()

def encode_categorical_ordinal(df: pd.DataFrame, column_names: list, order: list) -> pd.DataFrame:
    '''
    Encode categorical data
    args:
        df: pd.DataFrame
        column_names: list of column names
        handle_nan: how to handle nan values
    return:
        pd.DataFrame
    '''
    ordinal_encoder = preprocessing.OrdinalEncoder(categories=order, handle_unknown='use_encoded_value', unknown_value=-1)
    categorical_ordinal_df = df[column_names]
    # encode categorical ordinal data
    categorical_ordinal_df = ordinal_encoder.fit_transform(categorical_ordinal_df)
    # replace -1 with mean for each column
    for i in range(len(column_names)):
        mean = categorical_ordinal_df[categorical_ordinal_df[:, i] != -1, i].mean()
        categorical_ordinal_df[categorical_ordinal_df[:, i] == -1, i] = mean
    return categorical_ordinal_df

def encode_numerical(df: pd.DataFrame,
                     column_names: list,
                     scaling: str = 'standard',
                     handle_nan: str = 'mean') -> pd.DataFrame:
    '''
    Encode numerical data
    args:
        df: pd.DataFrame
        column_names: list of column names
        scaling: how to scale data (standard, minmax)
        handle_nan: how to handle nan values (mean, median, mode)
    return:
        pd.Dataframe
    '''
    numerical_df = df[column_names]
    # replace nan with mean for each column
    if handle_nan == 'mean':
        numerical_df.fillna(numerical_df.mean(), inplace=True)
    elif handle_nan == 'median':
        numerical_df.fillna(numerical_df.median(), inplace=True)
    elif handle_nan == 'mode':
        numerical_df.fillna(numerical_df.mode(), inplace=True)
    # scale data
    scaler = None
    if scaling == 'standard':
        scaler = preprocessing.StandardScaler()
    elif scaling == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    try:
        numerical_df = scaler.fit_transform(numerical_df)
    except ValueError:
        print('Error while scaling numerical data')
    except AttributeError:
        print('Scaler not found')
    return numerical_df

def prune_target(classes: list, labels: pd.DataFrame):
    # result in [charged off, fully paid]
    # generate a df that contain dropped entries
    labels = labels.copy()
    lookup_table = {x: idx for idx, x in enumerate(classes)}
    # generate a bool mask that mask out dropped entries
    mask = np.ones(len(labels), dtype=bool)
    mask[labels == lookup_table['Default']] = False
    mask[labels == lookup_table['Current']] = False
    mask[labels == lookup_table['Late (16-30 days)']] = False
    mask[labels == lookup_table['Late (31-120 days)']] = False
    mask[labels == lookup_table['In Grace Period']] = False
    # merge doesnt meet credit policy entries into fully paid and charged off
    
    labels[labels == lookup_table['Does not meet the credit policy. Status:Fully Paid']] = lookup_table['Fully Paid']
    labels[labels == lookup_table['Does not meet the credit policy. Status:Charged Off']] = lookup_table['Charged Off']
    # drop default
    labels = labels[labels != lookup_table['Default']]
    # drop current
    labels = labels[labels != lookup_table['Current']]
    # drop late
    labels = labels[labels != lookup_table['Late (16-30 days)']]
    labels = labels[labels != lookup_table['Late (31-120 days)']]
    # drop in grace period
    labels = labels[labels != lookup_table['In Grace Period']]
    uniques = np.unique(labels)
    reindex_class = []
    for u in uniques:
        reindex_class.append(classes[u])
    reindex_label = np.zeros(len(labels))
    reindex_label[labels == lookup_table['Fully Paid']] = 1
    reindex_label[labels == lookup_table['Charged Off']] = 0
    return reindex_class, reindex_label.astype(int), mask

def encode_target(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    '''
    Encode target data
    args:
        df: pd.DataFrame
        column_name: column name of target
    return:
        labels: pd.DataFrame
        classes: list of classes
        mask: bool mask that mask out dropped entries
    '''
    encoder = preprocessing.LabelEncoder()
    target_df = df[column_name]
    target_df = encoder.fit_transform(target_df)
    labels, classes, mask = prune_target(encoder.classes_, target_df)
    return labels, classes, mask

