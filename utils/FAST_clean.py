import pandas as pd
import os
from utils import data_utils


if __name__ == '__main__':
    dm_df = data_utils.load_all('raw trial data'+os.sep+'FAST-MAG_demog.csv')
    dm_df = dm_df[['INVNO', 'PTNO', 'VISIT', 'AGE', 'SEX']]
    dm_df = dm_df.rename(columns={'AGE':'onset_age', 'SEX':'GENDER_TX'})
    result_dm = dm_df.drop_duplicates(subset=['INVNO', 'PTNO']).rename(columns={'PTNO': 'CASE_ID'})
    data_utils.save_dataframe_to_csv(result_dm[['CASE_ID', 'onset_age', 'GENDER_TX']], 'FAST-dm')

    data_bi = data_utils.load_all('raw trial data'+os.sep+'FAST-MAG_barthel.csv')
    data_bi = data_bi[['INVNO', 'PTNO', 'VISIT', 'BFEED', 'BCHRBED', 'BGROOM', 'BTOILET', 'BBATH', 'BWALK', 'BSTAIRS',
                       'BDRESS', 'BBOWEL', 'BBLADDER']]
    data_bi = data_bi.rename(columns={'BCHRBED':'Transfers', 'BFEED':'Feeding', 'BGROOM':'Grooming', 'BTOILET':'Toilet_use',
                                     'BBATH':'Bathing', 'BWALK': 'Mobility', 'BSTAIRS':'Stairs', 'BDRESS':'Dressing',
                                     'BBOWEL':'Bowel_control', 'BBLADDER':'Bladder_control'})
    data_bi = data_bi.dropna().astype('int')
    data_bi['Barthel_Total'] = data_bi[['Feeding', 'Transfers', 'Bathing', 'Toilet_use', 'Grooming', 'Mobility',
                                        'Stairs', 'Dressing', 'Bowel_control', 'Bladder_control']].sum(axis=1)


    data_mrs = data_utils.load_all('raw trial data'+os.sep+'FAST-MAG_rankin.csv')
    data_mrs = data_mrs[['INVNO', 'PTNO', 'VISIT', 'MRANK']].rename(columns={'MRANK':'discharged_mrs'})
    data_mrs = data_mrs.dropna().astype('int')
    result_bm = pd.merge(data_bi, data_mrs, how='right', on=['INVNO', 'PTNO', 'VISIT']).dropna()
    result = pd.merge(dm_df, result_bm, how='right', on=['INVNO', 'PTNO']).dropna()


    result.insert(loc=0, column='CASE_ID', value=result.reset_index().index)
    result = result.drop(['INVNO', 'PTNO', 'VISIT_x', 'VISIT_y'], axis=1)
    data_utils.save_dataframe_to_csv(result, 'FAST')

    print('done')