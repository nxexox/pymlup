# Custom Python object as a REST API

## What this shows

MLup can expose a plain Python object as a FastAPI REST API - no ML framework and no
serialized model file required, only a `predict`-like method.

## Requirements

* Python 3.8+
* `pip install -r requirements.txt` (installs `pymlup`, no extra needed)

## Run

```bash
pip install -r requirements.txt
python run.py
```

`model.py` defines `MyModel`, a plain Python class with a `predict(self, X)` method. `run.py`
passes an instance of it straight to `mlup.UP(ml_model=...)` - no file, no pickling.

The API starts on `http://localhost:8009`. Interactive docs: `http://localhost:8009/docs`.

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
