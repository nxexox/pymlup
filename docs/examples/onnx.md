# Example: ONNX model

Serves an ONNX model with `mlup run`, including the one setting an ONNX model with typed
(float32) inputs needs.

Source: [`examples/onnx/`](https://github.com/nxexox/pymlup/tree/main/examples/onnx)

## Requirements

* Python 3.8+
* `pymlup[onnx]` to serve the model, plus `scikit-learn` and `skl2onnx` to prepare the example
  `model.onnx` file (not needed to serve it)

## Run

```bash
git clone https://github.com/nxexox/pymlup.git
cd pymlup/examples/onnx

pip install -r requirements.txt
bash run.sh
```

`run.sh` builds a tiny ONNX classifier ([`prepare_model.py`](https://github.com/nxexox/pymlup/blob/main/examples/onnx/prepare_model.py): trains a `DecisionTreeClassifier`, converts it with
`skl2onnx`) and starts the API with:

```bash
mlup run -m model.onnx --up.dtype_for_predict=float32
```

`--up.dtype_for_predict=float32` matters: `model.onnx` declares float32 inputs, but JSON numbers
decode to Python int/float (float64) by default, and onnxruntime rejects a dtype mismatch.

## Test API

```bash
curl -X POST http://localhost:8009/predict \
  -H "Content-Type: application/json" \
  -d '{"input_data": [[1, 1], [3, 3]]}'
```

## Expected result

```json
{"predict_result": [[0, 1], [[1.0, 0.0], [0.0, 1.0]]]}
```

The first element is the predicted class per row, the second is the per-class probability for
each row.

See the [full README for this example](https://github.com/nxexox/pymlup/blob/main/examples/onnx/README.md) on GitHub.
