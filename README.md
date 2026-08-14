![MLup logo](https://raw.githubusercontent.com/nxexox/pymlup/main/docs/assets/img/logo_with_name_blue.png)

----

[![Linters and testing](https://github.com/nxexox/pymlup/actions/workflows/python-package.yml/badge.svg)](https://github.com/nxexox/pymlup/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/pymlup.svg)](https://badge.fury.io/py/pymlup)
[![Downloads](https://img.shields.io/pypi/dm/pymlup.svg)](https://pypistats.org/packages/pymlup)

## Introduction

MLup turns Python objects and serialized machine learning models into self-hosted FastAPI REST APIs without writing serving boilerplate.

Serve sklearn, PyTorch, TensorFlow, ONNX, and custom Python models through a generated FastAPI application with validation, OpenAPI docs, and health endpoints.

* Pure Python, no required framework-specific glue code;
* Uses FastAPI for the web layer;
* Works with any Python object that exposes a `predict`-like method, plus native (de)serialization support for scikit-learn, lightgbm, tensorflow, torch and onnx models.

## Requirements

Python 3.10+ (3.11, 3.12, 3.13, 3.14 supported; the `tensorflow` extra requires Python <3.14 until TensorFlow publishes 3.14 wheels).

pymlup 0.4.0 runs on FastAPI and Pydantic v2. Python 3.8/3.9 and Pydantic v1 are still supported
on the pymlup 0.3.x line.

## Installation

```bash
pip install pymlup
```

With an ML backend extra:
```bash
pip install "pymlup[scikit-learn]"  # For scikit-learn
pip install "pymlup[lightgbm]"      # For microsoft lightgbm
pip install "pymlup[tensorflow]"    # For tensorflow
pip install "pymlup[torch]"         # For torch
pip install "pymlup[onnx]"          # For onnx models: torch, tensorflow, sklearn, etc...
```

## Quick start

From a clean environment to a working prediction API in about five minutes, using scikit-learn as the example.

**1. Create a virtual environment and install pymlup with the scikit-learn extra:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install "pymlup[scikit-learn]"
```

**2. Train and save a tiny model.** Save this as `train_model.py` and run it:
```python
# train_model.py
import pickle
from sklearn.tree import DecisionTreeClassifier

X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 0, 1, 1]

model = DecisionTreeClassifier().fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```
```bash
python train_model.py
```

**3. Start the API:**
```bash
mlup run -m ./model.pkl
```

**4. Check it's alive** (in another terminal):
```bash
curl http://localhost:8009/health
```
```json
{"status":200}
```

**5. Call the model:**
```bash
curl -X POST http://localhost:8009/predict \
  -H "Content-Type: application/json" \
  -d '{"X": [[1, 1], [3, 3]]}'
```
```json
{"predict_result":[0,1]}
```

**6. Explore the interactive API docs.** Open **http://localhost:8009/docs** in your browser for the Swagger UI, auto-generated from your model's `predict` signature — you can try `/predict` right there.

Stop the server with `Ctrl+C`.

Already have a serialized model (onnx, joblib, lightgbm, torch, tensorflow, ...) instead of training a new one? Point `mlup run -m` at it directly (install the matching extra from [Installation](#installation)) — see the [full documentation](https://mlup.org/) for every supported format and config option.

### Any Python object, no ML framework required

pymlup doesn't require a serialized model file at all — any Python object with a `predict`-like method works:
```python
import mlup

class MyAnyModelForExample:
    def predict(self, X):
        return X

up = mlup.UP(ml_model=MyAnyModelForExample())
up.ml.load()
up.run_web_app(daemon=True)
```
Open **http://localhost:8009/docs** to try it, or call it from the same script — this needs `pip install requests` separately, it's not a pymlup dependency:
```python
import requests
response = requests.post('http://localhost:8009/predict', json={'X': [[1, 2, 3], [4, 5, 6]]})
print(response.json())

up.stop_web_app()
```

## Supported ML frameworks

Work tested with machine learning model frameworks (links to tests):

* [scikit-learn>=1.2.0,<2.0.0](https://github.com/nxexox/pymlup/tree/main/tests/integration_tests/frameworks/test_scikit_learn_model.py)
* [tensorflow>=2.0.0,<3.0.0, Python<3.14](https://github.com/nxexox/pymlup/tree/main/tests/integration_tests/frameworks/test_tensorflow_model.py)
* [lightgbm>=4.0.0,<5.0.0](https://github.com/nxexox/pymlup/tree/main/tests/integration_tests/frameworks/test_lightgbm_model.py)
* [torch>=2.0.0,<3.0.0](https://github.com/nxexox/pymlup/tree/main/tests/integration_tests/frameworks/test_pytorch_model.py)
* [onnx>=1.0.0,<2.0.0](https://github.com/nxexox/pymlup/tree/main/tests/unit_tests/ml/test_binarization.py)
* [onnxruntime>=1.14.0,<1.26.0](https://github.com/nxexox/pymlup/tree/main/tests/unit_tests/ml/test_binarization.py)

Support and tested with machine learning libraries:

* [numpy>=1.0.0,<3.0.0](https://github.com/nxexox/pymlup/tree/main/tests/unit_tests/ml/test_data_transformers.py)
* [pandas>=2.0.0,<3.0.0](https://github.com/nxexox/pymlup/tree/main/tests/unit_tests/ml/test_data_transformers.py)
* [joblib>=1.2.0,<2.0.0](https://github.com/nxexox/pymlup/tree/main/tests/unit_tests/ml/test_binarization.py)

## Documentation

The full documentation — Python API, CLI reference, config file format, storages, binarizers, data transformers, web app architectures/API, application life cycle — lives at **[mlup.org](https://mlup.org/)** (source: [docs/](https://github.com/nxexox/pymlup/tree/main/docs)).

## Useful links
* [Full documentation](https://mlup.org/)
* [Examples](https://github.com/nxexox/pymlup/tree/main/examples)
* [Test models](https://github.com/nxexox/pymlup/tree/main/mldata)

## Metrics

MLup PyPi download statistics: https://pepy.tech/project/pymlup

[![Downloads](https://static.pepy.tech/badge/pymlup)](https://pepy.tech/project/pymlup)
[![Downloads](https://static.pepy.tech/badge/pymlup/month)](https://pepy.tech/project/pymlup)
[![Downloads](https://static.pepy.tech/badge/pymlup/week)](https://pepy.tech/project/pymlup)
