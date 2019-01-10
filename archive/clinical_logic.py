from utils import data_utils
import pandas as pd
import numpy as np


df = data_utils.load_all('tsr_mbn_raw.csv')
print(df.shape)
df['bi_total'] = pd.DataFrame(np.sum(df[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility',
                                         'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control']], axis=1))

df['nihss_total'] = pd.DataFrame(np.sum(df[['NIHS_1a_out', 'NIHS_1b_out', 'NIHS_1c_out', 'NIHS_2_out', 'NIHS_3_out', 'NIHS_4_out',
                                            'NIHS_5aL_out', 'NIHS_5bR_out', 'NIHS_6aL_out', 'NIHS_6bR_out', 'NIHS_7_out', 'NIHS_8_out',
                                            'NIHS_9_out', 'NIHS_10_out', 'NIHS_11_out']], axis=1))
# --
df = df[~((df['Mobility'] == 0) & (df['Stairs'] != 0))]
df = df[~((df['Stairs'] == 10) & (df['NIHS_6aL_out'] == 4) & (df['NIHS_6bR_out'] == 4))]
df = df[~((df['discharged_mrs'] != 5) & (df['NIHS_1a_out'] == 3))]
df = df[~(df['nihss_total'] > 39)]
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
df = df[~((df['bi_total'] == 0) & (df['discharged_mrs'] < 5))]
df.drop('bi_total', axis=1, inplace=True)
df.drop('nihss_total', axis=1, inplace=True)
print(df.shape)
data_utils.save_dataframe_to_csv(df, 'tsr_mbn')
print('done')