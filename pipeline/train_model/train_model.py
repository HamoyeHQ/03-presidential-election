import argparse
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_model(x_train, y_train):
    x_train_data = np.load(x_train, allow_pickle=True)
    y_train_data = np.load(y_train, allow_pickle=True)

    model = LogisticRegression()
    model.fit(x_train_data, y_train_data)

    joblib.dump(model, 'model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train')
    parser.add_argument('--y_train')
    args = parser.parse_args()
    train_model(args.x_train, args.y_train)
