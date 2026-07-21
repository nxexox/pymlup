![MLup logo](docs/assets/img/logo_with_name_blue.png?raw=true)

----

[![Linters and testing](https://github.com/nxexox/pymlup/actions/workflows/python-package.yml/badge.svg)](https://github.com/nxexox/pymlup/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/pymlup.svg)](https://badge.fury.io/py/pymlup)
[![Downloads](https://img.shields.io/pypi/dm/pymlup.svg)](https://pypistats.org/packages/pymlup)

## Introduction

MLup is a library for easy and fast running of ML models in production.

All you need is to deliver the model file to the server (a config is optional) — pymlup turns it into a FastAPI web application with one CLI command. No web app code to write or maintain.

* Pure Python, no required framework-specific glue code;
* Uses FastAPI for the web layer;
* Works with any Python object that exposes a `predict`-like method, plus native (de)serialization support for scikit-learn, lightgbm, tensorflow, torch and onnx models.

## Requirements

Python 3.8+ (3.12, 3.13, 3.14 supported; the `tensorflow` extra requires Python <3.14 until TensorFlow publishes 3.14 wheels).

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

The easiest way to try it, from a model file on disk:
```bash
mlup run -m /path/to/my/model.onnx
```

Or from Python, with any object that has a `predict`-like method:
```python
import mlup

class MyAnyModelForExample:
    def predict(self, X):
        return X

up = mlup.UP(ml_model=MyAnyModelForExample())
up.ml.load()
# You can open your browser at http://localhost:8009/docs for interactive API docs (Swagger UI)
up.run_web_app(daemon=True)

import requests
response = requests.post('http://0.0.0.0:8009/predict', json={'X': [[1, 2, 3], [4, 5, 6]]})
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

The full documentation — Python API, CLI reference, config file format, storages, binarizers, data transformers, web app architectures/API, application life cycle — lives at **[mlup.org](https://mlup.org/)** (source: [docs/](docs/)).

## Useful links
* [Full documentation](https://mlup.org/)
* [Examples](https://github.com/nxexox/pymlup/tree/main/examples)
* [Test models](https://github.com/nxexox/pymlup/tree/main/mldata)

## Metrics

MLup PyPi download statistics: https://pepy.tech/project/pymlup

[![Downloads](https://static.pepy.tech/badge/pymlup)](https://pepy.tech/project/pymlup)
[![Downloads](https://static.pepy.tech/badge/pymlup/month)](https://pepy.tech/project/pymlup)
[![Downloads](https://static.pepy.tech/badge/pymlup/week)](https://pepy.tech/project/pymlup)
