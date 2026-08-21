<p align="center">
  <img src="assets/img/logo_with_name_blue.png" alt="MLup logo" width="500">
</p>

# MLup

**MLup turns Python objects and serialized machine learning models into self-hosted FastAPI REST APIs without writing serving boilerplate.**

Serve sklearn, PyTorch, TensorFlow, ONNX, and custom Python models through a generated FastAPI application with validation, OpenAPI docs, and health endpoints.

[![Linters and testing](https://github.com/nxexox/pymlup/actions/workflows/python-package.yml/badge.svg)](https://github.com/nxexox/pymlup/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/pymlup.svg)](https://pypi.org/project/pymlup/)
[![Downloads](https://img.shields.io/pypi/dm/pymlup.svg)](https://pypistats.org/packages/pymlup)

<p align="center">
  <img src="assets/img/demo.gif" alt="MLup demo: mlup run, curl /predict, Swagger UI">
</p>

## Quick start

```bash
pip install "pymlup[scikit-learn]"
```

```python
# model.py
import pickle
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier().fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 0, 1, 1])
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
```

```bash
python model.py
mlup run -m model.pkl
```

```bash
curl -X POST http://localhost:8009/predict \
  -H "Content-Type: application/json" \
  -d '{"X": [[1, 1], [3, 3]]}'
# {"predict_result":[0,1]}
```

Interactive API docs: **http://localhost:8009/docs**. Full walkthrough and every option: [Quickstart](quickstart.md).

## What MLup provides

* A generated FastAPI application — `up.web.app` is a real `fastapi.FastAPI` instance;
* `POST /predict`, with request validation built from your model's signature or a column config;
* Auto-generated OpenAPI schema and Swagger UI at `/docs`;
* `GET /health` and `GET /info`;
* Loading from serialized models (pickle, joblib, and framework-native formats) or from a plain Python object with a `predict`-like method — `pip install pymlup` alone, no extra, covers the latter;
* Optional `worker_and_queue` and `batching` execution modes for the prediction call.

## Supported frameworks

| Type / framework | Typical formats | Installation extra |
| --- | --- | --- |
| Any Python object | in-memory object with a `predict`-like method — no file needed | *(none, core install)* |
| scikit-learn | pickle / joblib — via mlup's generic binarizers, not a dedicated adapter | `pymlup[scikit-learn]` |
| LightGBM | native LightGBM format, pickle, joblib | `pymlup[lightgbm]` |
| PyTorch | native torch formats (including TorchScript), pickle | `pymlup[torch]` |
| TensorFlow | SavedModel, `.h5`, `.keras`, pickle — Python <3.14 only | `pymlup[tensorflow]` |
| ONNX | `.onnx` | `pymlup[onnx]` |

See [Binarizers](binarizers.md) for how model-format detection works.

## When to use MLup

* You already have a trained scikit-learn, LightGBM, PyTorch, TensorFlow, or ONNX model, or a plain Python object with a `predict`-like method;
* You need an internal API or a small, self-hosted model service;
* You're moving a prototype out of a notebook and just need it reachable over HTTP;
* You don't need a full MLOps platform for this — just a clean serving layer.

## When MLup probably isn't the right fit

* Kubernetes-native autoscaling;
* Highly optimized multi-GPU inference;
* Distributed serving graphs;
* Managed cloud deployment;
* A full MLOps lifecycle / model registry;
* A complex, application-specific HTTP API with substantial custom logic beyond serving a model.

## Where to go next

* [Quick Start](quickstart.md)
* [Configuration file](config_file.md)
* [Python interface](python_interface.md)
* [Bash commands (CLI)](bash_commands.md)
* [Web app API](web_app_api.md)
* [Migrating from 0.3.x](migration/v0.4.md)
* [Examples](https://github.com/nxexox/pymlup/tree/main/examples) and [test models](https://github.com/nxexox/pymlup/tree/main/mldata) (GitHub)

## Downloads

MLup PyPI download statistics: https://pepy.tech/project/pymlup

[![Downloads](https://static.pepy.tech/badge/pymlup)](https://pepy.tech/project/pymlup)
[![Downloads](https://static.pepy.tech/badge/pymlup/month)](https://pepy.tech/project/pymlup)
[![Downloads](https://static.pepy.tech/badge/pymlup/week)](https://pepy.tech/project/pymlup)
