import pandas as pd
import os
from utils import data_utils




if __name__ == '__main__':
    bi_data_1 = data_utils.load_all('raw trial data'+os.sep+'ALIAS1_bi.csv').rename(columns = {'subjectid':'CASE_ID', 'F21Q01DAY':'DAY', 'Transfer':'Transfers',
                                                                     'Toilet use':'Toilet_use', 'Bowels':'Bowel_control',
                                                                     'Bladder':'Bladder_control',
                                                                     'F21TotalScore':'Barthel_Total'})
    mrs_data_1 = data_utils.load_all('raw trial data'+os.sep+'ALIAS1_mrs.csv').rename(columns = {'subjectid':'CASE_ID', 'F19Q01DAY':'DAY', 'F19Q02':'discharged_mrs'})

    dm_data_1 = data_utils.load_all('raw trial data'+os.sep+'ALIAS1_dm.csv')
    dm_data_1 = dm_data_1[['subjectid', 'F00zGender', 'F00age']]
    dm_data_1 = dm_data_1.rename(columns={'subjectid':'CASE_ID', 'F00age': 'onset_age', 'F00zGender': 'GENDER_TX'})

    result_1 = pd.merge(bi_data_1, mrs_data_1, how='right', on=['CASE_ID', 'DAY']).dropna()
    result_1 = pd.merge(dm_data_1, result_1, how='right', on=['CASE_ID']).dropna()


    bi_data_2 = data_utils.load_all('raw trial data'+os.sep+'ALIAS2_bi.csv').rename(columns = {'subjectid':'CASE_ID', 'F21Q01DAY':'DAY', 'Transfer':'Transfers',
                                                                     'Toilet use':'Toilet_use', 'Bowels':'Bowel_control',
                                                                     'Bladder':'Bladder_control',
                                                                     'F21TotalScore':'Barthel_Total'})
    mrs_data_2 = data_utils.load_all('raw trial data'+os.sep+'ALIAS2_mrs.csv').rename(columns = {'subjectid':'CASE_ID', 'F19Q01DAY':'DAY', 'Rankin Scale':'discharged_mrs'})

    dm_data_2 = data_utils.load_all('raw trial data'+os.sep+'ALIAS2_dm.csv')
    dm_data_2 = dm_data_2[['subjectid', 'F00zGender', 'F00Age']]
    dm_data_2 = dm_data_2.rename(columns={'subjectid':'CASE_ID', 'F00Age': 'onset_age', 'F00zGender': 'GENDER_TX'})

    result_2 = pd.merge(bi_data_2, mrs_data_2, how='right', on=['CASE_ID', 'DAY']).dropna().drop(['day90'], axis=1)
    result_2 = pd.merge(dm_data_2, result_2, how='right', on=['CASE_ID']).dropna()


    result = pd.concat([result_1, result_2], axis=0).drop(['DAY'], axis=1)
    data_utils.save_dataframe_to_csv(result, 'ALIAS')

    result_dm = result.drop_duplicates(subset=['CASE_ID'])
    data_utils.save_dataframe_to_csv(result_dm, 'ALIAS-dm')
    print('done')