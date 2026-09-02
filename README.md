<p align="center">
  <img src="https://raw.githubusercontent.com/nxexox/pymlup/main/docs/assets/img/logo_with_name_blue.png" alt="MLup logo" width="500">
</p>

**MLup turns Python objects and serialized machine learning models into self-hosted FastAPI REST APIs without writing serving boilerplate.**

Serve sklearn, PyTorch, TensorFlow, ONNX, and custom Python models through a generated FastAPI application with validation, OpenAPI docs, and health endpoints.

[![Linters and testing](https://github.com/nxexox/pymlup/actions/workflows/python-package.yml/badge.svg)](https://github.com/nxexox/pymlup/actions/workflows/python-package.yml)
[![PyPI version](https://img.shields.io/pypi/v/pymlup.svg)](https://pypi.org/project/pymlup/)
[![Downloads](https://img.shields.io/pypi/dm/pymlup.svg)](https://pypistats.org/packages/pymlup)

<p align="center">
  <img src="https://raw.githubusercontent.com/nxexox/pymlup/main/docs/assets/img/demo.gif" alt="MLup demo: mlup run, curl /predict, Swagger UI">
</p>

## Quick example

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
```
```json
{"predict_result":[0,1]}
```

Interactive API docs (Swagger UI): **http://localhost:8009/docs**

Full walkthrough, config options, and every supported model format: **[mlup.org/quickstart](https://mlup.org/quickstart/)**.

## What you get

* A generated FastAPI application — `up.web.app` is a real `fastapi.FastAPI` instance;
* `POST /predict`, with request validation built from your model's signature or a column config;
* Auto-generated OpenAPI schema and Swagger UI at `/docs`;
* `GET /health` and `GET /info`;
* Loading from serialized models (pickle, joblib, and framework-native formats) or from a plain Python object with a `predict`-like method — `pip install pymlup` alone, no extra, covers the latter;
* Optional `worker_and_queue` and `batching` execution modes for the prediction call.

## Why MLup

FastAPI isn't the problem — it's a solid, well-designed web framework. MLup exists because serving an existing model usually means rewriting the same boilerplate around it: a request schema, a `/predict` route, validation, a health check, an app object to run.

**Manual FastAPI:**
```python
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class PredictRequest(BaseModel):
    X: list

app = FastAPI()

@app.post("/predict")
def predict(body: PredictRequest):
    return {"predict_result": model.predict(body.X).tolist()}
```

**MLup:**
```bash
mlup run -m model.pkl
```

Use FastAPI directly when your API contains substantial custom application logic. Use MLup when you already have a model and mostly need a clean HTTP serving layer around it.

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

For those, look at [FastAPI](https://fastapi.tiangolo.com/) directly, or dedicated model-serving platforms like [BentoML](https://www.bentoml.com/), [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), [KServe](https://kserve.github.io/website/), or [Triton Inference Server](https://github.com/triton-inference-server/server).

## Supported models

| Type / framework | Typical formats | Installation extra |
| --- | --- | --- |
| Any Python object | in-memory object with a `predict`-like method — no file needed | *(none, core install)* |
| scikit-learn | pickle / joblib — via mlup's generic binarizers, not a dedicated adapter | `pymlup[scikit-learn]` |
| LightGBM | native LightGBM format, pickle, joblib | `pymlup[lightgbm]` |
| PyTorch | native torch formats (including TorchScript), pickle | `pymlup[torch]` |
| TensorFlow | SavedModel, `.h5`, `.keras`, pickle — Python <3.14 only | `pymlup[tensorflow]` |
| ONNX | `.onnx` | `pymlup[onnx]` |

Binarizer implementations: [`mlup/ml/binarization/`](https://github.com/nxexox/pymlup/tree/main/mlup/ml/binarization). Framework integration tests: [`tests/integration_tests/frameworks/`](https://github.com/nxexox/pymlup/tree/main/tests/integration_tests/frameworks).

## Examples

* [`examples/sklearn/`](https://github.com/nxexox/pymlup/blob/main/examples/sklearn/) — train, save, and serve a scikit-learn model, start to finish;
* [`examples/custom-python-model/`](https://github.com/nxexox/pymlup/blob/main/examples/custom-python-model/) — serve a plain Python object with a `predict`-like method, no ML framework;
* [`examples/onnx/`](https://github.com/nxexox/pymlup/blob/main/examples/onnx/) — serve an ONNX model;
* [`examples/from_config.py`](https://github.com/nxexox/pymlup/blob/main/examples/from_config.py) — load a saved YAML config and run it;
* [`examples/configs.py`](https://github.com/nxexox/pymlup/blob/main/examples/configs.py) — build a `mlup.Config` with explicit columns in code;
* [`examples/daemon.py`](https://github.com/nxexox/pymlup/blob/main/examples/daemon.py) — load a model from disk storage and expose the FastAPI `app` object (for `uvicorn`/`gunicorn`);
* [`examples/gunicorn_run.py`](https://github.com/nxexox/pymlup/blob/main/examples/gunicorn_run.py) — run under `uvicorn`/`gunicorn` instead of `mlup run`.

Browse all: [github.com/nxexox/pymlup/tree/main/examples](https://github.com/nxexox/pymlup/tree/main/examples).

## Documentation

Full documentation: **[mlup.org](https://mlup.org/)**

* [Quick Start](https://mlup.org/quickstart/)
* [Configuration file](https://mlup.org/config_file/)
* [Python interface](https://mlup.org/python_interface/)
* [Bash commands (CLI)](https://mlup.org/bash_commands/)
* [Migrating from 0.3.x](https://mlup.org/migration/v0.4/)

## Downloads

MLup PyPI download statistics: https://pepy.tech/project/pymlup

[![Downloads](https://static.pepy.tech/badge/pymlup)](https://pepy.tech/project/pymlup)
[![Downloads](https://static.pepy.tech/badge/pymlup/month)](https://pepy.tech/project/pymlup)
[![Downloads](https://static.pepy.tech/badge/pymlup/week)](https://pepy.tech/project/pymlup)

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](https://github.com/nxexox/pymlup/blob/main/CONTRIBUTING.md).

---

If MLup is useful to you, consider starring the repository — it helps other developers discover the project.
