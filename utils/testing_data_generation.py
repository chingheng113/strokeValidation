import numpy as np
import pandas as pd
from utils import data_utils
from random import randint

def basic_filter(df):
    df = df[~(df['discharged_mrs'] == 6)]
    # df = df[~((df['Stairs'] == 10) & (df['NIHS_6aL_out'] == 4) & (df['NIHS_6bR_out'] == 4))]
    # df = df[~(df['NIHSS_Total'] > 39)]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_1b_out'] != 2))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_1c_out'] != 2))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_4_out'] != 3))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_5aL_out'] != 4))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_5bR_out'] != 4))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_6aL_out'] != 4))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_6bR_out'] != 4))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_7_out'] != 0))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_8_out'] != 2))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_9_out'] != 3))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_10_out'] != 2))]
    # df = df[~((df['NIHS_1a_out'] == 3) & (df['NIHS_11_out'] != 0))]
    df = df[~((df['Barthel_Total'] == 0) & (df['discharged_mrs'] < 5))]
    # https://www.ahajournals.org/doi/abs/10.1161/01.str.30.8.1538
    df = df[~((df['Barthel_Total'] > 60) & (df['discharged_mrs'] > 3))]
    df = df[~((df['Barthel_Total'] < 59) & (df['discharged_mrs'] < 4))]
    return df


def get_id_bi_data(df):
    id_df = df['CASE_ID']
    bi_df = df[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility', 'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control']]
    return id_df, bi_df


def create_extreme_bi_outlier(df):
    df_normal = df[df['label'] != -1]
    columns = df_normal.columns.values
    indice = df_normal.index.values
    selected_columns = np.unique(np.random.choice(columns, randint(1, 10)))
    selected_index = np.random.choice(indice, randint(1, indice.size))
    for i in selected_index:
        row = df_normal.loc[[i]]
        for c in selected_columns:
            row[c]+= 20
        row['label'] = -1
        df = df.append(row)
    return df.drop_duplicates()


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
    # test_data = create_extreme_bi_outlier(test_data)
    data_utils.save_dataframe_to_csv(test_data, 'testing_'+str(mrs))
    return test_data


def do_transform(dataset, mrs, test_data):
    # training data
    id_df, bi_df, mrs_df, nih_df = data_utils.get_tsr(mrs, 'is')
    scaled_bi, scaler = data_utils.scale(bi_df)
    bi_df_pca, pca = data_utils.pca_reduction(bi_df)

    # testing data
    labels = test_data[['label']]
    test_bi = test_data.drop(['label'], axis=1)

    test_bi_scaled = scaler.transform(test_bi.values)
    test_bi_pca = pca.transform(test_bi_scaled)
    test_bi_pca_df = pd.DataFrame(data=test_bi_pca, index=test_data.index, columns=['pca_1', 'pca_2'])
    test_bi_pca_df['label'] = labels.values
    data_utils.save_dataframe_to_csv(test_bi_pca_df, dataset+'_testing_'+str(mrs)+'_pca')
    return test_bi_pca_df


if __name__ == '__main__':
    dataset = 'alias'
    if dataset == 'nih':
        df = data_utils.get_nih()
    elif dataset == 'alias':
        df = data_utils.get_alias()
    elif dataset == 'fast':
        df = data_utils.get_fast()
    elif dataset == 'tnk':
        df = data_utils.get_tnk()
    else:
        print('error')


    print(df.shape)
    df_clean = basic_filter(df)
    print(df_clean.shape)
    data_utils.save_dataframe_to_csv(df_clean, 'aa.csv')
    df_0 = df_clean[df_clean['discharged_mrs'] == 0]
    print(df_0.shape)
    df_1 = df_clean[df_clean['discharged_mrs'] == 1]
    print(df_1.shape)
    df_2 = df_clean[df_clean['discharged_mrs'] == 2]
    print(df_2.shape)
    df_3 = df_clean[df_clean['discharged_mrs'] == 3]
    print(df_3.shape)
    df_4 = df_clean[df_clean['discharged_mrs'] == 4]
    print(df_4.shape)
    df_5 = df_clean[df_clean['discharged_mrs'] == 5]
    print(df_5.shape)

    mixed_0 = mix_bi_data(0, df_0, df_4, df_5)
    do_transform(dataset, 0, mixed_0)

    mixed_1 = mix_bi_data(1, df_1, df_4, df_5)
    do_transform(dataset, 1, mixed_1)

    mixed_2 = mix_bi_data(2, df_2, df_0, df_5)
    do_transform(dataset, 2, mixed_2)

    mixed_3 = mix_bi_data(3, df_3, df_0, df_5)
    do_transform(dataset, 3, mixed_3)

    mixed_4 = mix_bi_data(4, df_4, df_0, df_1)
    do_transform(dataset, 4, mixed_4)

    mixed_5 = mix_bi_data(5, df_5, df_0, df_1)
    do_transform(dataset, 5, mixed_5)

    print('done')