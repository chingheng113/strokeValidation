import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import data_utils


def bubble_plot(data, group_names):
    data_size = data.groupby(group_names).size()
    keys = data.groupby(group_names).groups.keys()
    x =[]
    y =[]
    for key in keys:
        x.append(key[0])
        y.append(key[1])
    plt.title('Bubble plot of TSR (mRS vs. BI)')
    plt.scatter(x, y, s=data_size, alpha=.5)
    plt.xlabel(group_names[0])
    plt.ylabel(group_names[1])
    plt.show()


def violin_plot(data):
    sns.violinplot(data.iloc[: ,0], data.iloc[:, 1], orient='v')
    sns.despine()
    plt.show()


def see_plot(data):
    data = data[data['discharged_mrs'] == 1]
    # bi_totoal = data.iloc[: ,1]
    # plt.hist(bi_totoal, bins='auto')
    # data["Barthel_Total"].apply(np.log).hist()
    plt.show()

if __name__ == '__main__':
    df = data_utils.get_ischemic('TSR_2017.csv')
    # df = data_utils.get_nih()
    n = 'NIHSS_Total'
    b = 'Barthel_Total'
    m = 'discharged_mrs'
    df_nbm = df[[n, b, m]]
    df_nbm = df_nbm[~((df_nbm.discharged_mrs == 0) & (df_nbm.Barthel_Total == 0))]
    see_plot(df_nbm[[m, b]])
    # bubble_plot(df_nbm[[m, b]], [m, b])
    # violin_plot(df_nbm[[m, b]])