"""Generate the lightgbm test fixtures in mldata/models/.

Standalone replacement for the old lightgbm.ipynb notebook (which had the
model filenames misspelled as "ligthgbm-*" instead of "lightgbm-*", not
matching what tests/conftest.py actually expects).

Run from the mldata/ directory: python generate_lightgbm_model.py
"""
import csv
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

COLUMNS = [
    'RainToday', 'MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm',
]


def main():
    with open('datasets/weatherAUS.csv') as f:
        reader = csv.reader(f)
        data = list(reader)
        df = pd.DataFrame(data[1:], columns=data[0])

    df = df[df[COLUMNS] != 'NA'].dropna(subset=COLUMNS)
    X, Y = df.loc[:, df.columns != 'RainToday'], df.loc[:, df.columns == 'RainToday']
    X[COLUMNS[1:]] = X[COLUMNS[1:]].astype({col: float for col in COLUMNS[1:]})
    Y = Y.RainToday.map({'Yes': 1, 'No': 0})

    gbm = lgb.train(
        {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2', 'l1'},
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'seed': 42,
        },
        lgb.Dataset(X[COLUMNS[1:]], Y),
        num_boost_round=20,
        valid_sets=lgb.Dataset(X[COLUMNS[1:]], Y),
        callbacks=[lgb.early_stopping(stopping_rounds=5)],
    )

    test_input = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float64)
    pred = gbm.predict(test_input, num_iteration=gbm.best_iteration)
    print('Test predict:', pred, 'rounded:', round(float(pred[0]), 4))

    gbm.save_model('models/lightgbm-binary_cls_model.txt')
    with open('models/lightgbm-binary_cls_model.pckl', 'wb') as f:
        pickle.dump(gbm, f)


if __name__ == '__main__':
    main()
