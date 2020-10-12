import argparse
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def test_model(x_test, y_test, model_path):
    x_test_data = np.load(x_test, allow_pickle=True)
    y_test_data = np.load(y_test, allow_pickle=True)

    model = joblib.load(model_path)
    y_pred = model.predict(x_test_data)

    report = classification_report(y_test_data, y_pred)

    with open('output.txt', 'a') as f:
        f.write(str(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_test')
    parser.add_argument('--y_test')
    parser.add_argument('--model')
    args = parser.parse_args()
    test_model(args.x_test, args.y_test, args.model)
