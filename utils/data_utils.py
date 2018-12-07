import os
import csv
import pandas as pd
import numpy as np
from sklearn import preprocessing as sp
from sklearn import decomposition



data_source_path = 'data_source'+os.sep


def get_file_path(file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + data_source_path)
    return os.path.join(filepath + file_name)


def save_array_to_csv(file_name, title, patients_dic):
    write_file_path = get_file_path(file_name+'.csv')
    with open(write_file_path, 'w') as write_csv:
        w = csv.DictWriter(write_csv, title)
        w.writeheader()
        for d in patients_dic.keys():
            p_dic = patients_dic[d]
            w.writerow(p_dic)


def save_dataframe_to_csv(df, file_name):
    dirname = os.path.dirname(__file__)
    filepath = os.path.join(dirname, '..' + os.sep + data_source_path)
    df.to_csv(filepath + file_name + '.csv', sep=',', index=False)


def load_all(fn):
    read_file_path = get_file_path(fn)
    df = pd.read_csv(read_file_path, encoding='utf8')
    # sampling
    # df = df.groupby('discharged_mrs', group_keys=False).apply(lambda x: x.sample(300)).sample(frac=1)
    return df


def get_tsr(target, subtype):
    if subtype == 'is':
        df = get_ischemic('TSR.csv')
    elif subtype == 'he':
        df = get_hemorrhagic('TSR.csv')
    else:
        df = load_all('TSR.csv')
    if target != '':
        df = df[df['discharged_mrs'] == target]
    id_df = df.iloc[:, 0:1]
    bi_df = df.iloc[:, 3:13]
    mrs_df = df.iloc[:, 14:15]
    nih_df = df.iloc[:, 15:30]
    return id_df, bi_df, mrs_df, nih_df


def get_ischemic(fn):
    df = load_all(fn)
    return df[(df['ICD_ID_1.0'] == 1) | (df['ICD_ID_2.0'] == 1)]


def get_hemorrhagic(fn):
    df = load_all(fn)
    return df[(df['ICD_ID_3.0'] == 1) | (df['ICD_ID_4.0'] == 1)]


def get_nih():
    df = load_all('NIH.csv')
    return df


def get_nih_test(mrs):
    df = load_all('testing_'+str(mrs)+'.csv')
    return df


def scale(x_data):
    # scaled_data = np.round(sp.MinMaxScaler(feature_range=(0, 1)).fit_transform(x_data), 3)
    scaled_data = np.round(sp.StandardScaler().fit_transform(x_data), 3)
    scaled_df = pd.DataFrame(scaled_data, index=x_data.index, columns=x_data.columns)
    return scaled_df


def pca_reduction(data):
    pca = decomposition.PCA(n_components=2)
    scaled_data = scale(data)
    pca.fit(scaled_data)
    transformed_data = pca.transform(scaled_data)
    df_pca = pd.DataFrame(transformed_data, columns=['pca_1', 'pca_2'], index=data.index)
    return df_pca, pca


def label_data(df, df_unique, labels):
    df_unique['label'] = labels
    all_label = pd.DataFrame(index=df.index, columns=['label'])
    for index_unique, row_unique in df_unique.iterrows():
        row = df.loc[[index_unique]]
        all_row = df.where(df.values == row.values).dropna()
        i = all_row.index
        all_label.loc[i] = row_unique['label']
    df['label'] = all_label.values
    return df, df_unique


def outlier_filter(df, df_unique):
    outliers_unique = df_unique[df_unique.label == -1]
    outliers_all = df[df.label == -1]
    return outliers_unique, outliers_all



if __name__ == '__main__':

    print('Done')