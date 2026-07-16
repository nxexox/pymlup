"""Generate the scikit-learn test fixtures in mldata/models/.

Standalone replacement for the old scikit-learn.ipynb notebook, so the
fixtures can be regenerated with a single command against whatever
scikit-learn version is currently pinned in pyproject.toml.

Note: this intentionally does NOT regenerate scikit-learn-binary_cls_model.onnx.
skl2onnx (last released 1.20.0) can't convert a DecisionTreeClassifier trained
with a current scikit-learn (it emits a boolean where ONNX expects an int
attribute, a skl2onnx/sklearn version-compat gap upstream). The already-committed
.onnx file still loads and predicts correctly under current onnxruntime, so it's
left as-is; only the .pckl is regenerated here.

Run from the mldata/ directory: python generate_sklearn_model.py
"""
import csv
import pickle

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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

    model = DecisionTreeClassifier(max_depth=100, random_state=42)
    model.fit(X[COLUMNS[1:]], Y)

    test_input = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.float32)
    print('Test predict:', model.predict(test_input))

    with open('models/scikit-learn-binary_cls_model.pckl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
