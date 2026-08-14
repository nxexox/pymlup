# Quickstart

This page walks through the fastest path from a clean environment to a working prediction API, then covers other ways to run and configure mlup.

## Requirements

* Python 3.10+
* pymlup 0.4.x uses Pydantic v2

## Fast run

**1. Install pymlup with the scikit-learn extra:**
```bash
pip install "pymlup[scikit-learn]"
```

**2. Train and save a tiny model**, e.g. in `train_model.py`:
```python
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

**3. Run it:**
```bash
mlup run -m ./model.pkl
```

**4. Check it's alive and call it** (in another terminal):
```bash
curl http://localhost:8009/health
# {"status":200}

curl -X POST http://localhost:8009/predict \
  -H "Content-Type: application/json" \
  -d '{"X": [[1, 1], [3, 3]]}'
# {"predict_result":[0,1]}
```

**5. Open the interactive API docs** at http://localhost:8009/docs (Swagger UI, auto-generated from your model's `predict` signature) — you can try `/predict` right there.

Stop the server with `Ctrl+C`, or with `up.stop_web_app()` if you started it from Python (see below).

You can also load a model that's already in a Python variable instead of a file on disk — for example directly in a Jupyter notebook:
```python
import mlup

model: YourModel

up = mlup.UP(ml_model=model)
up.ml.load()
up.run_web_app(daemon=True)
# Testing your web application
up.stop_web_app()
```

You can pass your settings directly to the bash command:
```bash
mlup run -m /path/to/your/model.pckl --up.port=8011
```
or in your code:
```python
import mlup

model: YourModel

up = mlup.UP(ml_model=model, conf=mlup.Config(port=8011))
up.ml.load()
up.run_web_app(daemon=True)
# Testing your web application
up.stop_web_app()
```

You can read about all the settings at [description of the configuration file](config_file.md).

You can check how the data is processed, the model makes a prediction, and the response is processed without launching the web application.
There are methods for this:

* `UP.predict` - Method that is called by the web application with the data received in the request.
If the `auto_detect_predict_params=True` flag is set in the config (See [Config: auto_detect_predict_params](config_file.md#model-load-settings)), the arguments of this method are the same as the arguments of the model's predict method.
If the `auto_detect_predict_params=False` flag is set in the config, the data is passed in the `data_for_predict` argument.
* `UP.async_predict` - Asynchronous version of `UP.predict`.
* `UP.predict_from` - Same as `UP.predict`, but does not call the data transformer before calling the model predictor.
This allows you to quickly test the model, without transforming your test data into a valid JSON format.
  
```python
import numpy
import mlup

class MyModel:
    def predict(self, X):
        return X

model = MyModel()

up = mlup.UP(ml_model=model, conf=mlup.Config(auto_detect_predict_params=True))
up.ml.load()

obj_1 = [1, 2, 3]
obj_2 = [4, 5, 6]
objs_for_predict = [obj_1, obj_2]
up.predict(X=objs_for_predict)
await up.async_predict(X=objs_for_predict)
up.predict_from(X=numpy.array(objs_for_predict))

up.conf.auto_detect_predict_params = False
# Refresh mlup model settings
up.ml.load_model_settings()

up.predict(data_for_predict=objs_for_predict)
await up.async_predict(data_for_predict=objs_for_predict)
up.predict_from(data_for_predict=numpy.array(objs_for_predict))
```

## Different model types

By default, mlup calls the model's `predict` method.
This behavior can be changed using the `predict_method_name="predict"` parameter.
For models that are callable, `predict_method_name="__call__"` should be specified.
For example, for `tensorflow`, `torch` models.

```python
import mlup
from mlup.ml.empty import EmptyModel

up = mlup.UP(ml_model=EmptyModel(), conf=mlup.Config(predict_method_name="__call__"))
```

Also, models can be binarized in different ways: `pickle`, `joblib`, `onnx`, etc.
By default, `binarization_type="auto"`: mlup inspects the file's content and extension and picks the most likely binarizer for you automatically.
This behavior can be changed by specifying the `binarization_type` parameter. You can specify one of the [mlup binarizers](https://github.com/nxexox/pymlup/tree/main/mlup/ml/binarization/) or specify your own (See [Binarizers](binarizers.md)).

## Launch on servers

There is one important difference between the local configuration and the server configuration.
On the server, the model is always loaded from storage - for example, from a local disk. On the local, you can load the model directly from a variable.

_P.S. When you pickle `mlup.UP` of an object, the model is saved along with the `mlup.UP` object and is not additionally loaded from disk._

To do this, you need to specify the path to the model on the server in the config. Two parameters are responsible for this: `storage_type` and `storage_kwargs`.

```python
import mlup
from mlup import constants

up = mlup.UP(
    conf=mlup.Config(
        storage_type=constants.StorageType.disk,
        storage_kwargs={
            'path_to_files': '/path/to/your/model/on/server.extension',
            'files_mask': 'server.extension',
        },
    )
)
```

mlup creates an object from `storage_type` and uses `storage_kwargs` as creation arguments.
In the case of `mlup.constants.StorageType.disk`, you must specify the path to the `path_to_files` model and can specify `files_mask`.
`files_mask` is a regular expression that will find your model in `path_to_files`.

By default, mlup `storage_type=constants.StorageType.memory`.
