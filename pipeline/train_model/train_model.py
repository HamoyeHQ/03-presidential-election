import argparse

def train(data_path):
  import pickle
  import sys 
  import os
  import joblib
  import numpy as np
  #from sklearn.tree import DecisionTreeRegressor
  from sklearn.ensemble import ExtraTreesRegressor
    
 
    
  X_train = np.load(X_train, allow_pickle=True)
  y_train = np.load(y_train, allow_pickle=True)

 

  r = ExtraTreesRegressor(n_estimators=400, random_state=42)
  r.fit(X_train, y_train.ravel())

  

    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()
    train_model(args.X_train, args.y_train)
