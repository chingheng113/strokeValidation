import pandas as pd
from utils import data_utils


if __name__ == '__main__':
    data_bi = data_utils.load_all('p_barthel.csv')
    data_bi = data_bi[['INVNO', 'PTNO', 'VISIT', 'BFEED', 'BCHRBED', 'BGROOM', 'BTOILET', 'BBATH', 'BWALK', 'BSTAIRS',
                       'BDRESS', 'BBOWEL', 'BBLADDER', 'BARTHT']]
    data_bi = data_bi.rename(columns={'BCHRBED':'Transfers', 'BFEED':'Feeding', 'BGROOM':'Grooming', 'BTOILET':'Toilet_use',
                                     'BBATH':'Bathing', 'BWALK': 'Mobility', 'BSTAIRS':'Stairs', 'BDRESS':'Dressing',
                                     'BBOWEL':'Bowel_control', 'BBLADDER':'Bladder_control', 'BARTHT':'Barthel_Total'})
    data_mrs = data_utils.load_all('p_rankin.csv')
    data_mrs = data_mrs[['INVNO', 'PTNO', 'VISIT', 'MRANK']].rename(columns={'MRANK':'discharged_mrs'})
    result = pd.merge(data_bi, data_mrs, how='right', on=['INVNO', 'PTNO', 'VISIT']).dropna()
    result.insert(loc=0, column='CASE_ID', value=result.reset_index().index)
    result = result.drop(['INVNO', 'PTNO', 'VISIT'], axis=1)
    data_utils.save_dataframe_to_csv(result, 'FAST')
    print('done')