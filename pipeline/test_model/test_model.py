
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def test (data_path):
  import pickle
  import argparse
  import joblib
  import os
  import sys, subprocess;
  subprocess.run([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
  import numpy as np
  from sklearn.metrics import mean_absolute_error as MAE
  from sklearn.metrics import mean_squared_error as MSE
  from sklearn import metrics
  from sklearn.metrics import r2_score
  from sklearn.ensemble import ExtraTreesRegressor
    

  #Load saved model
  with open(f'{data_path}/model','rb') as file:
    model = pickle.load(file)
    
  X_test = np.load(X_test, allow_pickle=True)
  y_test = np.load(y_test, allow_pickle=True)



  y_predET = model.predict(X_test)

  print('Mean Absolute Error: ', round(metrics.mean_absolute_error(y_test, y_predET), 3))
  print('Mean Squared Error: ', round(metrics.mean_squared_error(y_test, y_predET), 3))
  #print('Root Mean Squared Error: ', round(np.sqrt(metrics.mean_squared_error(y_test, y_predET)), 3))
  print('R2 score: ', round(r2_score(y_test, y_predET), 3))

  #save result
  with open(f'{data_path}/model_result', 'wb') as result:
    pickle.dump(y_predET, result)

    #with open('output.txt', 'a') as f:
        #f.write(str(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--X_test')
    parser.add_argument('--y_test')
    parser.add_argument('--model')
    args = parser.parse_args()
    test_model(args.X_test, args.y_test, args.model)
