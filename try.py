from utils import data_utils

dataset = 'alias'
if dataset == 'nih':
    df = data_utils.get_nih()
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
    df = data_utils.get_tsr('', '')
else:
    print('error')

df = df[df.discharged_mrs != 6]
print(df_dm.shape[0])
print(df_dm.onset_age.mean())
print(df_dm.onset_age.std())
print(df_dm[df_dm.GENDER_TX == 2].shape)
print(df_dm[df_dm.GENDER_TX == 2].shape[0]/df_dm.shape[0])

print('--')
print(df[df.discharged_mrs == 4].shape[0])
print(df[df.discharged_mrs == 4].shape[0]/df.shape[0])

a = df[(df.Barthel_Total > 59) & (df.Barthel_Total < 80)]
print(a.shape[0])
print(a.shape[0]/df.shape[0])