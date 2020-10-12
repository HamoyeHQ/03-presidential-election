import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

def _preprocess_data():
    '''
    This function/component will:
    1. Load the presidential elections dataset
    2. Clean and transform the dataset
    3. Split the dataset into train and test set
    4. Use np.save to save our dataset to disk so that it can be reused by later components
    '''
    dataset_path='./president-1976-2016.csv'
    df = pd.read_csv(dataset_path, error_bad_lines=False)
    df.drop(['office','notes','version','writein','state_ic', 'state_cen','state_po','state_fips'], axis = 1, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['year','candidate', 'state'], inplace=True)

    candidate = [i for i in df.candidate]
    year = df.year.tolist()
    df.candidate = [candidate[i] + '_' + str(year[i]) for i in range(len(year))]

    df['candidatevotes_propotion'] = df['candidatevotes']/df['totalvotes']

    #Transform the dataset from long format to a wide format
    df_wide = df.pivot(index='candidate', columns='state', values='candidatevotes_propotion').reset_index()
    df_wide.fillna(0, inplace=True)

    candidate_party = df[['candidate', 'party']].drop_duplicates(subset=['candidate']).reset_index().drop('index', axis=1)

    df2 = pd.merge(df_wide, candidate_party, how='inner', on=['candidate'])

    party = []
    for party_name in df2.party:
        if party_name.lower() != 'democrat' or party_name.lower() != 'republican':
            party.append('others')
        else:
            party.append(party_name)

    df2['party'] = party

    parties = {'democrat': 1, 'republican': 2, 'others': 3}
    df2['party'] = df2['party'].replace(parties)

    #list of candiatates that have won presidential elections since 1976
    presidents = ['Carter, Jimmy_1976', 'Reagan, Ronald_1980', 'Reagan, Ronald_1984', 'Bush, George H.W._1988',
                  'Clinton, Bill_1992', 'Clinton, Bill_1996', 'Bush, George W._2000', 'Bush, George W._2004',
                  'Obama, Barack H._2008', 'Obama, Barack H._2012', 'Trump, Donald J._2016']

    #Create a target variable
    df2['target'] = df2['candidate'].isin(presidents).astype(int)

    X, y= df2.drop(columns=['candidate','target']), df2['target']

    sm = SMOTE(random_state=42)
    X, target = sm.fit_sample(X, y.ravel())

    rfc = RandomForestClassifier(random_state=101)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X, target)

    X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.3, random_state=0)

    np.save('x_train.npy', X_train)
    np.save('x_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

if __name__ == '__main__':
    print('Preprocessing presidential elections data...')
    _preprocess_data()
