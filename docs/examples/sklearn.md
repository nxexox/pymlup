# Example: scikit-learn model

Trains a tiny scikit-learn model, saves it to disk, and serves it as a REST API with a single
`mlup run` command - no serving code written.

Source: [`examples/sklearn/`](https://github.com/nxexox/pymlup/tree/main/examples/sklearn)

## Requirements

* Python 3.8+
* `pymlup[scikit-learn]`

## Run

```bash
git clone https://github.com/nxexox/pymlup.git
cd pymlup/examples/sklearn

pip install -r requirements.txt
bash run.sh
```

`run.sh` trains a `DecisionTreeClassifier` ([`train.py`](https://github.com/nxexox/pymlup/blob/main/examples/sklearn/train.py)), saves it to `model.pkl`, and starts the API with
`mlup run -m model.pkl`.

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

See the [full README for this example](https://github.com/nxexox/pymlup/blob/main/examples/sklearn/README.md) on GitHub.
