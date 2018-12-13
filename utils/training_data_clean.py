import numpy as np
import pandas as pd
from utils import data_utils

def basic_filter(df):
    df = df[~(df['discharged_mrs'] == 6)]
    df = df[~((df['Barthel_Total'] == 0) & (df['discharged_mrs'] < 5))]
    # https://www.ahajournals.org/doi/abs/10.1161/01.str.30.8.1538
    df = df[~((df['Barthel_Total'] > 60) & (df['discharged_mrs'] > 3))]
    df = df[~((df['Barthel_Total'] < 59) & (df['discharged_mrs'] < 4))]
    return df

if __name__ == '__main__':
    df = data_utils.load_all('TSR.csv')
    print(df.shape)
    df = basic_filter(df)
    data_utils.save_dataframe_to_csv(df, 'TSR_cleaned')
    print(df.shape)