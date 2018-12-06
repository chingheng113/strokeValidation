import numpy as np
import pandas as pd
from utils import data_utils

def basic_filter(df):
    df = df[~(df['discharged_mrs'] == 6)]
    df = df[~((df['Stairs'] == 10) & (df['NIHS_6aL_out'] == 4) & (df['NIHS_6bR_out'] == 4))]
    df = df[~(df['NIHSS_Total'] > 39)]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_1b_out'] != 2))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_1c_out'] != 2))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_4_out'] != 3))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_5aL_out'] != 4))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_5bR_out'] != 4))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_6aL_out'] != 4))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_6bR_out'] != 4))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_7_out'] != 0))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_8_out'] != 2))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_9_out'] != 3))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_10_out'] != 2))]
    df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_11_out'] != 0))]
    df = df[~((df['Barthel_Total'] == 0) & (df['discharged_mrs'] < 5))]
    print(df.shape)
    return df

#def mix_data(df_base, df_add_1, df_add_2):


if __name__ == '__main__':
    nih_df = data_utils.get_nih()
    nih_df_clean = basic_filter(nih_df)
    nih_df_0 = nih_df_clean[nih_df_clean['discharged_mrs'] == 0]
    print(nih_df_0.shape)
    nih_df_1 = nih_df_clean[nih_df_clean['discharged_mrs'] == 1]
    print(nih_df_1.shape)
    nih_df_2 = nih_df_clean[nih_df_clean['discharged_mrs'] == 2]
    print(nih_df_2.shape)
    nih_df_3 = nih_df_clean[nih_df_clean['discharged_mrs'] == 3]
    print(nih_df_3.shape)
    nih_df_4 = nih_df_clean[nih_df_clean['discharged_mrs'] == 4]
    print(nih_df_4.shape)
    nih_df_5 = nih_df_clean[nih_df_clean['discharged_mrs'] == 5]
    print(nih_df_5.shape)

    mix_data(nih_df_0, nih_df_4, nih_df_5)
    # nih_df_final = nih_df_clean[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility',
    #                             'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control', 'Barthel_Total', 'discharged_mrs']]
    # data_utils.save_dataframe_to_csv(nih_df_final, 'testing')
    print('done')