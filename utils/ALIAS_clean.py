import pandas as pd
from utils import data_utils




if __name__ == '__main__':
    bi_data_1 = data_utils.load_all('ALIAS1_bi.csv').rename(columns = {'subjectid':'CASE_ID', 'F21Q01DAY':'DAY', 'Transfer':'Transfers',
                                                                     'Toilet use':'Toilet_use', 'Bowels':'Bowel_control',
                                                                     'Bladder':'Bladder_control',
                                                                     'F21TotalScore':'Barthel_Total'})
    mrs_data_1 = data_utils.load_all('ALIAS1_mrs.csv').rename(columns = {'subjectid':'CASE_ID', 'F19Q01DAY':'DAY', 'F19Q02':'discharged_mrs'})
    result_1 = pd.merge(bi_data_1, mrs_data_1, how='right', on=['CASE_ID', 'DAY']).dropna()

    bi_data_2 = data_utils.load_all('ALIAS2_bi.csv').rename(columns = {'subjectid':'CASE_ID', 'F21Q01DAY':'DAY', 'Transfer':'Transfers',
                                                                     'Toilet use':'Toilet_use', 'Bowels':'Bowel_control',
                                                                     'Bladder':'Bladder_control',
                                                                     'F21TotalScore':'Barthel_Total'})
    mrs_data_2 = data_utils.load_all('ALIAS2_mrs.csv').rename(columns = {'subjectid':'CASE_ID', 'F19Q01DAY':'DAY', 'Rankin Scale':'discharged_mrs'})
    result_2 = pd.merge(bi_data_2, mrs_data_2, how='right', on=['CASE_ID', 'DAY']).dropna().drop(['day90'], axis=1)


    result = pd.concat([result_1, result_2], axis=0).drop(['DAY'], axis=1)
    data_utils.save_dataframe_to_csv(result, 'ALIAS')
    print('done')