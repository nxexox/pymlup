# Example: custom Python model

Exposes a plain Python object as a REST API - no ML framework and no serialized model file,
only a `predict`-like method.

Source: [`examples/custom-python-model/`](https://github.com/nxexox/pymlup/tree/main/examples/custom-python-model)

## Requirements

* Python 3.8+
* `pymlup` (no extra needed)

## Run

```bash
git clone https://github.com/nxexox/pymlup.git
cd pymlup/examples/custom-python-model

pip install -r requirements.txt
python run.py
```

[`model.py`](https://github.com/nxexox/pymlup/blob/main/examples/custom-python-model/model.py)
defines `MyModel`, a plain Python class with a `predict(self, X)` method.
[`run.py`](https://github.com/nxexox/pymlup/blob/main/examples/custom-python-model/run.py) passes
an instance of it straight to `mlup.UP(ml_model=...)` - no file, no pickling.

## Test API

```bash
curl -X POST http://localhost:8009/predict \
  -H "Content-Type: application/json" \
  -d '{"X": [[1, 2], [3, 4]]}'
```

## Expected result

```json
{"predict_result": [3, 7]}
```

See the [full README for this example](https://github.com/nxexox/pymlup/blob/main/examples/custom-python-model/README.md) on GitHub.
