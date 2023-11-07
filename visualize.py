from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def visualize_nan(df: pd.DataFrame):
    '''
    Visualize nan values across each threshold
    args:
        df: pd.DataFrame
    '''
    threshold = [0.2, 0.4, 0.6, 0.8]
    column_at_threshold = []
    for t in threshold:
        # count number of column with above threshold nan
        nan_columns = df.isna().sum() > t*len(df)
        percentage = nan_columns.sum()/len(df.columns)
        column_at_threshold.append(percentage)
    # plot
    plt.bar(threshold, column_at_threshold, width=0.1)
    plt.xlim(0, 1)
    plt.xlabel('Threshold')
    plt.ylabel('Percentage of columns')
    plt.title('Percentage of columns with nan values exceeding threshold')

    

    