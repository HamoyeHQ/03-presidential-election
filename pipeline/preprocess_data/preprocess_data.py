def preprocess (data_path):
   
  import pickle
  import os
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
    
  # reading the dataset from the csv file
  df_new = pd.read_csv("https://raw.githubusercontent.com/sophiabj/g05-used-cars/master/data/new_vehicle.csv")
  # selecting features, X
  X = df_new.iloc[:, :-1].values
  # selecting labels, y
  y = df_new.iloc[:, -1].values

  # normalize the data
  X = StandardScaler().fit_transform(X.astype(float))

  # to split the data
  # split into train and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  #Save output file to path
 
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

if __name__ == '__main__':
    print('Done preprocessing...')
    _preprocess_data()
