import pandas as pd
from utils import data_utils

if __name__ == '__main__':
    data_bi = data_utils.load_all('TNK-barthelindex.csv')
    data_bi = data_bi[['VisitID', 'PtID', 'Feeding', 'Transfer', 'Grooming', 'Toilet use', 'Bathing', 'Mobility_a',
                       'Mobility_b', 'Stairs', 'Dressing', 'Bowels', 'Bladder', 'CalcTotalStore']].rename(columns={
                        'Transfer':'Transfers', 'Toilet use':'Toilet_use', 'Bowels':'Bowel_control',
                        'Bladder':'Bladder_control', 'CalcTotalStore':'Barthel_Total'})
    data_bi = data_bi[((data_bi.Mobility_a == -9) & (data_bi.Mobility_b != -9)) | ((data_bi.Mobility_a != -9) & (data_bi.Mobility_b == -9))]
    data_bi.insert(loc=data_bi.columns.get_loc("Mobility_a"), column='Mobility', value=data_bi['Mobility_b']+data_bi['Mobility_a']+9)
    data_bi = data_bi.drop(['Mobility_a', 'Mobility_b'], axis=1)
    data_mrs = data_utils.load_all('TNK-mrs.csv')
    data_mrs = data_mrs[['VisitID', 'PtID', 'MRankinScale']].rename(columns={'MRankinScale':'discharged_mrs'})

    result = pd.merge(data_bi, data_mrs, how='right', on=['VisitID', 'PtID']).dropna().rename(columns={'PtID':'CASE_ID'}).drop(['VisitID'], axis=1)
    data_utils.save_dataframe_to_csv(result, 'TNK')
    print('done')