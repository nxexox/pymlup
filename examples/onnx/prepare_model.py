"""Train a tiny scikit-learn model and convert it to ONNX format.

This step needs scikit-learn and skl2onnx (see requirements.txt). Serving the
resulting model.onnx with mlup does not - mlup only needs the `onnx` extra.

No external dataset, no network access.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)
y = [0, 0, 1, 1]

model = DecisionTreeClassifier().fit(X, y)

onnx_model = convert_sklearn(
    model,
    initial_types=[("X", FloatTensorType([None, 2]))],
    # zipmap=False keeps both outputs (labels and probabilities) as plain
    # arrays instead of a list of label->probability dicts.
    options={id(model): {"zipmap": False}},
)

with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Saved model.onnx")
