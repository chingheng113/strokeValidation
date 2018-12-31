import numpy as np
import pandas as pd
from utils import data_utils
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma


def basic_filter(df):
    df = df[~(df['discharged_mrs'] == 6)]
    df = df[~((df['Barthel_Total'] == 0) & (df['discharged_mrs'] < 4))]
    # df = df[~((df['Barthel_Total'] < 90) & (df['discharged_mrs'] == 0))]
    # https://www.ahajournals.org/doi/abs/10.1161/01.str.30.8.1538
    # df = df[~((df['Barthel_Total'] > 60) & (df['discharged_mrs'] > 3))]
    # df = df[~((df['Barthel_Total'] < 59) & (df['discharged_mrs'] < 4))]
    return df


def iqr(df):
    df_sub = df[df.discharged_mrs == 3]
    bi = df_sub.Barthel_Total
    print(bi.describe())
    Q1 = bi.quantile(0.25)
    Q3 = bi.quantile(0.75)
    IQR = Q3 - Q1
    print(Q1 - 1.5 * IQR)
    print(Q3 + 1.5 * IQR)


def frequency_outlier(df):
    mrs = 0
    df_sub = df[df.discharged_mrs == mrs]
    frequency = df_sub.groupby('Barthel_Total').size()
    frequency_log = np.log(frequency)
    a = frequency_log.mean()+frequency_log.std()
    w = np.min(frequency_log[frequency_log>a].index.values)

    print(w)
    print('--')
    print(frequency_log.describe())
    print()


def hist_log(df, mrs):
    df_sub = df[df.discharged_mrs == mrs]
    plt.style.use('seaborn-white')
    bins = range(0, 105, 5)
    n, bins, patches = plt.hist(df_sub.Barthel_Total, bins=bins, log=True)
    plt.title('mRS='+str(mrs))
    plt.xlabel('Total of Barthel index')
    plt.xticks(bins)
    plt.show()


def hist_org(df, mrs):
    df_sub = df[df.discharged_mrs == mrs]
    plt.style.use('seaborn-white')
    bins = range(0, 105, 5)
    n, bins, patches = plt.hist(df_sub.Barthel_Total, bins=bins)
    plt.title('mRS='+str(mrs))
    plt.xlabel('Total of Barthel index')
    plt.xticks(bins)
    plt.show()


def hist_break(df, mrs):
    df_sub = df[df.discharged_mrs == mrs]
    f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    bins = range(0, 105, 5)
    ax.hist(df_sub.Barthel_Total, bins=bins)
    ax2.hist(df_sub.Barthel_Total, bins=bins)

    ax.set_ylim(2000, 10000)  # outliers only
    ax2.set_ylim(0, 200)  # most of the data
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    ax.set_title('mRS='+str(mrs))
    plt.xlabel('Total of Barthel index')
    plt.xticks(bins)
    plt.show()


def boxplot(df):
    bi = []
    for i in range(0, 6):
        bi_sub = df[df.discharged_mrs == i].Barthel_Total
        bi.append(bi_sub)
        Q1 = bi_sub.quantile(0.25)
        Q3 = bi_sub.quantile(0.75)
        IQR = Q3 - Q1
        print(IQR)
        print(Q1 - 1.5 * IQR)
        print(Q3 + 1.5 * IQR)
    plt.boxplot(bi, positions=range(0, 6))
    plt.xlabel('mRS')
    plt.ylabel('BI total')
    plt.show()


if __name__ == '__main__':
    df = data_utils.load_all('TSR_2017.csv')
    print(df.shape)
    df = basic_filter(df)
    # boxplot(df)
    mrs = 0
    hist_org(df, mrs)
    hist_break(df, mrs)
    # hist_log(df, mrs)
    # df = iqr(df)
    # data_utils.save_dataframe_to_csv(df, 'TSR_2017_cleaned')
