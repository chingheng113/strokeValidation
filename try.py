from utils import data_utils

dataset = 'tnk'
if dataset == 'nih':
    df = data_utils.get_nih()
elif dataset == 'alias':
    df = data_utils.get_alias()
elif dataset == 'fast':
    df = data_utils.get_fast()
elif dataset == 'tnk':
    df = data_utils.get_tnk()
elif dataset == 'tsr':
    df = data_utils.get_tsr('', '')
else:
    print('error')

df = df[df.discharged_mrs != 6]
# print(df.onset_age.mean())
# print(df.onset_age.std())
# print(df[df.GENDER_TX == 'M'].shape)
# print(df[df.GENDER_TX == 'M'].shape[0]/df.shape[0])
print(df[df.discharged_mrs == 4].shape[0])
print(df[df.discharged_mrs == 4].shape[0]/df.shape[0])

a = df[(df.Barthel_Total > -1) & (df.Barthel_Total < 20)]
print(a.shape[0])
print(a.shape[0]/df.shape[0])