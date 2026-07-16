"""Generate the tensorflow test fixtures in mldata/models/.

Standalone replacement for the old tensorflow.ipynb / tensorflow-ONLY-37.ipynb
notebooks, so the fixtures can be regenerated with a single command against
whatever tensorflow version is currently pinned in pyproject.toml.

Note: this intentionally does NOT produce tensorflow-binary_cls_model.onnx.
tf2onnx (last released 1.17.0) can't introspect a Keras 3 model's internal
tensor naming (KeyError on the output tensor lookup), a tf2onnx/keras
version-compat gap upstream. No test fixture in tests/conftest.py actually
loads that .onnx file (confirmed by grepping the test suite), so it's simply
not generated rather than shipping a broken artifact.

Run from the mldata/ directory: python generate_tensorflow_model.py
"""
import csv
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

COLUMNS = [
    'RainToday', 'MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm',
]


def main():
    tf.random.set_seed(42)

    with open('datasets/weatherAUS.csv') as f:
        reader = csv.reader(f)
        data = list(reader)
        df = pd.DataFrame(data[1:], columns=data[0])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    df = df[df[COLUMNS] != 'NA'].dropna(subset=COLUMNS)
    X, Y = df.loc[:, df.columns != 'RainToday'], df.loc[:, df.columns == 'RainToday']
    X[COLUMNS[1:]] = X[COLUMNS[1:]].astype({col: float for col in COLUMNS[1:]})
    Y = Y.RainToday.map({'Yes': 1, 'No': 0})

    model.fit(X[COLUMNS[1:]], Y, epochs=5)

    test_input = np.array([[1., 2., 3., 4., 5., 6., 7., 8.]])
    print('Test predict:', model(test_input))

    with open('models/tensorflow-binary_cls_model.pckl', 'wb') as f:
        pickle.dump(model, f)

    model.save('models/tensorflow-binary_cls_model.keras')
    model.save('models/tensorflow-binary_cls_model.h5')
    model.export('models/tensorflow-binary_cls_model.savedmodel')


if __name__ == '__main__':
    main()
