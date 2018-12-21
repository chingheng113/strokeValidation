import numpy as np
import pandas as pd
from utils import data_utils

def basic_filter(df):
    df = df[~(df['discharged_mrs'] == 6)]
    df = df[~((df['Barthel_Total'] == 0) & (df['discharged_mrs'] < 4))]
    # https://www.ahajournals.org/doi/abs/10.1161/01.str.30.8.1538
    # df = df[~((df['Barthel_Total'] > 60) & (df['discharged_mrs'] > 3))]
    # df = df[~((df['Barthel_Total'] < 59) & (df['discharged_mrs'] < 4))]
    return df


def iqr(df):
    df_sub = df[df.discharged_mrs == 0]
    bi = df_sub.Barthel_Total
    print(bi.describe())
    Q1 = bi.quantile(0.25)
    Q3 = bi.quantile(0.75)
    IQR = Q3 - Q1
    print(Q1 - 1.5 * IQR)
    print(Q3 + 1.5 * IQR)


def frequency_outlier(df):
    df_sub = df[df.discharged_mrs == 0]
    frequency = df_sub.groupby('Barthel_Total').size()
    frequency_log = np.log(frequency)

    print(frequency.describe())
    print(frequency.quantile(0.95))

if __name__ == '__main__':
    df = data_utils.load_all('TSR_2017.csv')
    print(df.shape)
    df = basic_filter(df)
    frequency_outlier(df)
    # df = iqr(df)
    # data_utils.save_dataframe_to_csv(df, 'TSR_2017_cleaned')
