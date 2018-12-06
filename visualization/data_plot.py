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

if __name__ == '__main__':
    df = data_utils.get_ischemic('TSR.csv')
    n = 'NIHSS_Total'
    b = 'Barthel_ Total'
    m = 'discharged_mrs'
    df_nbm = df[[n, b, m]]
    bubble_plot(df_nbm[[m, b]], [m, b])
    violin_plot(df_nbm[[m, b]])