# ONNX model as a REST API

## What this shows

Serving an ONNX model with `mlup run`, including the one setting (`dtype_for_predict`) an ONNX
model with typed (float32) inputs needs.

## Requirements

* Python 3.8+
* `pip install -r requirements.txt` - installs `pymlup[onnx]` (needed to *serve* the model) plus
  `scikit-learn` and `skl2onnx` (needed only to *prepare* the example `model.onnx` file below;
  serving it does not need either)

## Run

```bash
pip install -r requirements.txt
bash run.sh
```

`run.sh` builds a tiny ONNX classifier (`prepare_model.py`: trains a `DecisionTreeClassifier`,
converts it with `skl2onnx`) and starts the API with `mlup run -m model.onnx`. To run the steps
yourself:

```bash
python prepare_model.py
mlup run -m model.onnx --up.dtype_for_predict=float32
```

`--up.dtype_for_predict=float32` matters: `model.onnx` declares float32 inputs, but JSON numbers
decode to Python int/float (float64) by default, and onnxruntime rejects a dtype mismatch.

The API starts on `http://localhost:8009`. Interactive docs: `http://localhost:8009/docs`.

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
each row - both are exactly what the ONNX model's two outputs (`output_label`,
`output_probability`) produce, unfiltered.
