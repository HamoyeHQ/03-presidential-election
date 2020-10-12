import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler

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
    df.drop(['notes','state_po','candidate', 'office','state','writein','state_ic'], axis = 1,inplace=True )
    df.dropna(inplace=True)

    X = df.drop('party', axis=1)
    target = df['party']

    rfc = RandomForestClassifier(random_state=101)
    rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
    rfecv.fit(X, target)

    X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)
    y = target

    #split the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    #Scaling the data
    scaler = MinMaxScaler()
    normalised_train_df = scaler.fit_transform(x_train)
    normalised_train_df = pd.DataFrame(normalised_train_df, columns=x_train.columns)

    normalised_test_df = scaler.transform(x_test)
    normalised_test_df = pd.DataFrame(normalised_test_df, columns=x_train.columns)

    #split the data
    X_train, X_test = normalised_train_df, normalised_test_df
    np.save('x_train.npy', X_train)
    np.save('x_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

if __name__ == '__main__':
    print('Preprocessing presidential elections data...')
    _preprocess_data()
