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


def get_id_bi_data(df):
    id_df = df['CASE_ID']
    bi_df = df[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility', 'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control']]
    return id_df, bi_df


def mix_bi_data(mrs, base, mix1, mix2):
    base_id, base_bi = get_id_bi_data(base)
    base_bi_unique = base_bi.drop_duplicates()

    mix1_id, mix1_bi = get_id_bi_data(mix1)
    mix1_label = pd.DataFrame(index=mix1.index, columns=['label'])

    mix2_id, mix2_bi = get_id_bi_data(mix2)
    mix2_label = pd.DataFrame(index=mix2.index, columns=['label'])

    for inx, row in mix1_bi.iterrows():
        isin = base_bi_unique.where(base_bi_unique.values == mix1_bi.loc[[inx]].values).dropna()
        if isin.empty:
            mix1_label.loc[inx] = -1
        else:
            mix1_label.loc[inx] = 0
    mix1_bi_label = pd.concat([mix1_bi, mix1_label], axis=1)

    for inx, row in mix2_bi.iterrows():
        isin = base_bi_unique.where(base_bi_unique.values == mix2_bi.loc[[inx]].values).dropna()
        if isin.empty:
            mix2_label.loc[inx] = -1
        else:
            mix2_label.loc[inx] = 0
    mix2_bi_label = pd.concat([mix2_bi, mix2_label], axis=1)

    base_bi['label'] = 0
    test_data = pd.concat([base_bi, mix1_bi_label, mix2_bi_label], axis=0).drop_duplicates()
    data_utils.save_dataframe_to_csv(test_data, 'testing_'+str(mrs))
    print('stop')

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

    mix_bi_data(0, nih_df_0, nih_df_4, nih_df_5)
    mix_bi_data(1, nih_df_1, nih_df_4, nih_df_5)
    mix_bi_data(2, nih_df_2, nih_df_0, nih_df_5)
    mix_bi_data(3, nih_df_3, nih_df_0, nih_df_5)
    mix_bi_data(4, nih_df_4, nih_df_0, nih_df_1)
    mix_bi_data(5, nih_df_5, nih_df_0, nih_df_1)

    print('done')