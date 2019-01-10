from utils import data_utils

dataset = 'tsr'
if dataset == 'nih':
    df = data_utils.get_nih()
    df_dm = df.drop_duplicates(subset=['CASE_ID'], keep=False)
elif dataset == 'alias':
    df = data_utils.get_alias()
    df_dm = data_utils.load_all('ALIAS-dm.csv')
elif dataset == 'fast':
    df = data_utils.get_fast()
    df_dm = data_utils.load_all('FAST-dm.csv')
elif dataset == 'tnk':
    df = data_utils.get_tnk()
    df_dm = data_utils.load_all('TNK-dm.csv')
elif dataset == 'tsr':
    df = data_utils.load_all('TSR_2017_cleaned.csv')
    df['id'] = df.CASE_ID.str.split('-').str[1]
    df_dm = df.drop_duplicates(subset=['id'], keep=False)
else:
    print('error')

df = df[df.discharged_mrs != 6]
print('number:', df_dm.shape[0])
print('age:', round(df_dm.onset_age.mean(),2))
print('sd:', round(df_dm.onset_age.std(),2))

print('fe', df_dm[df_dm.GENDER_TX == 0].shape[0])
print(df_dm[df_dm.GENDER_TX == 0].shape[0]/df_dm.shape[0])

print('--')
print(df[df.discharged_mrs == 4].shape[0])
print(df[df.discharged_mrs == 4].shape[0]/df.shape[0])

a = df[(df.Barthel_Total > 59) & (df.Barthel_Total < 80)]
print(a.shape[0])
print(a.shape[0]/df.shape[0])