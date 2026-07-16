"""Generate the pytorch test fixtures in mldata/models/.

Standalone replacement for the old pytorch.ipynb notebook, so the fixtures
can be regenerated with a single command against whatever torch version is
currently pinned in pyproject.toml.

Run from the mldata/ directory: python generate_pytorch_model.py
"""
import csv
import pickle

import numpy as np
import pandas as pd
import torch as tr
import torch.nn as nn

COLUMNS = [
    'RainToday', 'MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm',
]


def main():
    tr.manual_seed(42)

    with open('datasets/weatherAUS.csv') as f:
        reader = csv.reader(f)
        data = list(reader)
        df = pd.DataFrame(data[1:], columns=data[0])

    model = nn.Sequential(
        nn.Linear(8, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )
    loss_fn = nn.BCELoss()
    optimizer = tr.optim.Adam(model.parameters(), lr=1e-3)

    df = df[df[COLUMNS] != 'NA'].dropna(subset=COLUMNS)
    X, Y = df.loc[:, df.columns != 'RainToday'], df.loc[:, df.columns == 'RainToday']
    X[COLUMNS[1:]] = X[COLUMNS[1:]].astype({col: float for col in COLUMNS[1:]})
    Y = Y.RainToday.map({'Yes': 1, 'No': 0})

    X_tensor = tr.tensor(X[COLUMNS[1:]].values)
    Y_tensor = tr.tensor(Y.values)

    model.train()
    for batch, (_x, _y) in enumerate(zip(X_tensor, Y_tensor)):
        _x, _y = _x.to('cpu'), _y.to('cpu').float()

        pred = model(_x.float())
        loss = loss_fn(pred[0], _y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            current = batch + 1
            print(f'loss: {loss.item():>7f}  [{current:>5d}/{X_tensor.shape[0]:>5d}]')

    model.eval()
    test_input = tr.tensor(np.array([[1., 2., 3., 4., 5., 6., 7., 8.]])).to('cpu').float()
    print('Test predict:', model(test_input))

    tr.save(model, 'models/pytorch-binary_cls_model.pth')

    model_scripted = tr.jit.script(model)
    model_scripted.save('models/pytorch-binary_cls_model-jit.pth')

    with open('models/pytorch-binary_cls_model.pckl', 'wb') as f:
        pickle.dump(model, f)

    tr.onnx.export(
        model,
        tr.tensor(np.array([[1., 2., 3., 4., 5., 6., 7., 8.]], dtype=np.float32)),
        'models/pytorch-binary_cls_model.onnx',
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        external_data=False,
    )

    import onnxruntime as ort
    ort_sess = ort.InferenceSession('models/pytorch-binary_cls_model.onnx')
    outputs = ort_sess.run(None, {'input': np.array([[1., 2., 3., 4., 5., 6., 7., 8.]], dtype=np.float32)})
    print('ONNX test predict:', outputs)


if __name__ == '__main__':
    main()
