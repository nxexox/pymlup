# sklearn model as a REST API

## What this shows

Turning a trained scikit-learn model into a REST API with a single `mlup run` command, no
serving code written.

## Requirements

* Python 3.8+
* `pip install -r requirements.txt` (installs `pymlup[scikit-learn]`)

## Run

```bash
pip install -r requirements.txt
bash run.sh
```

`run.sh` trains a tiny `DecisionTreeClassifier` (`train.py`), saves it to `model.pkl`, and starts
the API with `mlup run -m model.pkl`. To run the steps yourself:

```bash
python train.py
mlup run -m model.pkl
```

The API starts on `http://localhost:8009`. Interactive docs: `http://localhost:8009/docs`.

## Test API

```bash
curl -X POST http://localhost:8009/predict \
  -H "Content-Type: application/json" \
  -d '{"X": [[1, 1], [3, 3]]}'
```

## Expected result

```json
{"predict_result": [0, 1]}
```
