from utils import data_utils
import pandas as pd

dataset = 'tsr'
if dataset == 'nih':
    df = data_utils.get_nih()
    df_dm = df.drop_duplicates(subset=['CASE_ID'])
elif dataset == 'alias':
    df = data_utils.get_alias()
    df_nu = df.drop_duplicates(subset=['CASE_ID'])
    df_dm = data_utils.load_all('ALIAS-dm.csv')
    df_dm = pd.merge(df_dm, df_nu, how='inner', on='CASE_ID').dropna()
    df_dm = df_dm.rename(columns={'onset_age_x':'onset_age', 'GENDER_TX_x':'GENDER_TX', 'discharged_mrs_x':'discharged_mrs',
                          'Barthel_Total_x':'Barthel_Total'})
elif dataset == 'fast':
    df = data_utils.get_fast()
    df_nu = df.drop_duplicates(subset=['CASE_ID']).drop(['onset_age', 'GENDER_TX'], axis=1)
    df_dm = data_utils.load_all('FAST-dm.csv')
    df_dm = pd.merge(df_dm, df_nu, how='inner', on='CASE_ID').dropna()
elif dataset == 'tnk':
    df = data_utils.get_tnk()
    df_nu = df.drop_duplicates(subset=['CASE_ID'])
    df_dm = data_utils.load_all('TNK-dm.csv')
    df_dm = pd.merge(df_dm, df_nu, how='inner', on='CASE_ID').dropna()
    print('')
elif dataset == 'tsr':
    df = data_utils.load_all('TSR_2017_lowess.csv')
    df['id'] = df.CASE_ID.str.split('-').str[1]
    df_dm = df.drop_duplicates(subset=['id'])
else:
    print('error')

df = df[df.discharged_mrs != 6]
print('number:', df_dm.shape[0])
print('age:', round(df_dm.onset_age.mean(),1))
print('sd:', round(df_dm.onset_age.std(),1))

print('fe', df_dm[df_dm.GENDER_TX == 2].shape[0])
print('fe%', round((df_dm[df_dm.GENDER_TX == 2].shape[0]/df_dm.shape[0]),3))

print('ma', df_dm[df_dm.GENDER_TX == 1].shape[0])
print('ma%', round((df_dm[df_dm.GENDER_TX == 1].shape[0]/df_dm.shape[0]),3))

print('--')
print('mrs 0:', df[df.discharged_mrs == 0].shape[0])
print('mrs 0:', round((df[df.discharged_mrs == 0].shape[0]/df.shape[0]),3))
print('--')
print('mrs 1:', df[df.discharged_mrs == 1].shape[0])
print('mrs 1:', round((df[df.discharged_mrs == 1].shape[0]/df.shape[0]),3))
print('--')
print('mrs 2:', df[df.discharged_mrs == 2].shape[0])
print('mrs 2:', round((df[df.discharged_mrs == 2].shape[0]/df.shape[0]),3))
print('--')
print('mrs 3:', df[df.discharged_mrs == 3].shape[0])
print('mrs 3:', round((df[df.discharged_mrs == 3].shape[0]/df.shape[0]),3))
print('--')
print('mrs 4:', df[df.discharged_mrs == 4].shape[0])
print('mrs 4:', round((df[df.discharged_mrs == 4].shape[0]/df.shape[0]),3))
print('--')
print('mrs 5:', df[df.discharged_mrs == 5].shape[0])
print('mrs 5:', round((df[df.discharged_mrs == 5].shape[0]/df.shape[0]),3))
print('--')
print('b1', df[(df.Barthel_Total > 79) & (df.Barthel_Total < 101)].shape[0])
print('b1', round((df[(df.Barthel_Total > 79) & (df.Barthel_Total < 101)].shape[0]/df.shape[0]), 3))
print('--')
print('b2', df[(df.Barthel_Total > 59) & (df.Barthel_Total < 80)].shape[0])
print('b2', round((df[(df.Barthel_Total > 59) & (df.Barthel_Total < 80)].shape[0]/df.shape[0]), 3))
print('--')
print('b3', df[(df.Barthel_Total > 39) & (df.Barthel_Total < 60)].shape[0])
print('b3', round((df[(df.Barthel_Total > 39) & (df.Barthel_Total < 60)].shape[0]/df.shape[0]), 3))
print('--')
print('b4', df[(df.Barthel_Total > 19) & (df.Barthel_Total < 40)].shape[0])
print('b4', round((df[(df.Barthel_Total > 19) & (df.Barthel_Total < 40)].shape[0]/df.shape[0]), 3))
print('--')
print('b5', df[(df.Barthel_Total > -1) & (df.Barthel_Total < 20)].shape[0])
print('b5', round((df[(df.Barthel_Total > -1) & (df.Barthel_Total < 20)].shape[0]/df.shape[0]), 3))