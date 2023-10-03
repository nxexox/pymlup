# Description of the Python Interface

You can use all the features of mlup directly in python code.

To do this, you need to import mlup into your environment.

```python
import mlup
```

After that, you can create as many `mlup.UP` applications as you like.

```python
import mlup

up1 = mlup.UP()
up2 = mlup.UP()
# ...
```

## mlup.UP

The `mlup.UP` object can be created in different ways:
* Create models in the variable `mlup.UP(ml_model=your_model)`
* Create from the config `mlup.UP.load_from_yaml(path_ti_your_conf)`
* Create with empty model:
```python
import mlup
from mlup.ml.empty import EmptyModel

up = mlup.UP(ml_model=EmptyModel())
```

You can also specify your own config when creating:
```python
import mlup
from mlup.ml.empty import EmptyModel

up = mlup.UP(
    ml_model=EmptyModel(),
    conf=mlup.Config(
        # ...
    )
)
```

`mlup.UP` has methods:
* `load_from_dict` - Creating a `mlup.UP` object from a dictionary with a config.
* `load_from_yaml` - Creating a `mlup.UP` object from a file with yaml config.
* `load_from_json` - Creating a `mlup.UP` object from a file with a json config.
* `to_dict` - Returns the config of an existing `mlup.UP` in the form of a dictionary.
* `to_yaml` - Saves the existing `mlup.UP` config to a yaml file.
* `to_json` - Saves the existing `mlup.UP` config to a json file.
* `predict` - Calls model prediction `mlup.UP.ml.predict` on the running event_loop.
* `predict_from` - Same as `predict`, but without data processing before prediction.
* `async async_predict` - Asynchronous version of `predict`. Fires in your event_loop.
* `run_web_app` - Runs a web application.
* `stop_web_app` - Stops a running web application.

And also, there are properties:
* `mlup.UP.ml` is a wrapper over the ml model. It contains all the logic for working with the model and processing data for it.
* `mlup.UP.web` is a wrapper for the web application. This object contains all the logic for creating, configuring and operating a web application.

You can change the configuration while `mlup.UP` is alive, without having to recreate the `mlup.UP` object.
```python
import mlup
from mlup.ml.empty import EmptyModel

up = mlup.UP(
    ml_model=EmptyModel(),
    conf=mlup.Config(
        # ...
    )
)
up.ml.load()

up.conf.use_thread_loop = False
up.ml.load(force_loading=True)
```

## mlup.UP.ml

The `mlup.UP.ml` object is `mlup.ml.model.MLupModel`.

`mlup.UP.ml` is a wrapper around your ml model. Other than the `mlup.UP.ml.load` method, in most cases you won't need to access it directly.

`mlup.UP.ml` has methods:
* `load` - a method that loads the model into memory, analyzes it and prepares the `MLupModel` object for working with the model.
* `load_model_settings` - only parses the loaded model and configures the internals of `MLupModel` after parsing. Called inside `load`.
* `get_X_from_predict_data` - searches and extracts from the data for prediction, the main argument with features X, according to the results of the analysis of the loaded model.
* `async predict` - causes model prediction, along with data processing.
* `async predict_from` - the same as `predict`, but without processing before prediction.

You can read about scenarios for using the `load`, `load_model_settings`, `get_X_from_predict_data` methods in [Description of the application life cycle](https://github.com/nxexox/pymlup/tree/main/docs/life_cycle.md#mlupml).

The `predict` and `predict_from` methods accept data for prediction based on the keys for which you send it to the model.
```python
import numpy
import mlup
from sklearn.tree import DecisionTreeClassifier

x = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [3, 4, 5], [5, 6, 7]])
y = numpy.array([1, 0, 1, 1, 0, 1])
model = DecisionTreeClassifier()
model.fit(x, y)
# Sklearn model have X, how main argument for features
print("Predict directly from ml model")
print(model.predict(X=x))

up = mlup.UP(ml_model=model)
up.ml.load()
print("Predict from mlup.UP")
print(up.predict(X=x.tolist()))
print("Predict from mlup.UP from numpy")
print(up.predict_from(X=x))
```

_P.S. The web application uses the `predict` method to call a prediction for the client._

You can also check whether the model has been loaded and parsed by using `loaded`.
```python
import mlup
from mlup.ml.empty import EmptyModel


up = mlup.UP(ml_model=EmptyModel())
print("Before mlup.UP.ml.load()")
print(up.ml.loaded)

up.ml.load()
print("After mlup.UP.ml.load()")
print(up.ml.loaded)

up.conf.use_thread_loop = False
print("After mlup.UP.ml.load() and change model config")
print(up.ml.loaded)
```

If you need to access your model, you can do so through the `model_obj` property.
```python
import mlup
from mlup.errors import ModelLoadError
from mlup.ml.empty import EmptyModel

model = EmptyModel()
up = mlup.UP(ml_model=model)
print("Before mlup.UP.ml.load()")
try:
    m = up.ml.model_obj
except ModelLoadError:
    print("Model not loaded for get source model")

up.ml.load()
print("After mlup.UP.ml.load()")
print(up.ml.model_obj)
print(model is up.ml.model_obj)
```

## mlup.UP.web

The `mlup.UP.web` object is `mlup.web.app.MLupWebApp`.

`mlup.UP.web` is a wrapper around a FastAPI web application. In most cases, you won't need to contact it directly.

`mlup.UP.web` has methods:
* `load` - a method that creates a web application, according to the config and the results of the analysis of `mlup.ml.model.MLupModel.load()`.
* `load_web_app_settings` - prepares some internal configs for launching the web application. Called inside `load`.
* `run` - Launches the web application.
* `stop` - Stops a running web application.
* `async http_health` - This is a handler for the `/health` request.
* `async info` - This is the `/info` request handler in case of `debug=False`.
* `async debug_info` - This is the `/info` request handler in the case of `debug=True`.
* `async http_health` - This is the `/predict` request handler.

You can read about scenarios for using the `load`, `load_web_app_settings`, `run`, `stop` methods in [Description of the application life cycle](https://github.com/nxexox/pymlup/tree/main/docs/life_cycle.md#mlupweb).

```python
import numpy
import mlup
import requests
from sklearn.tree import DecisionTreeClassifier
import time

x = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4], [3, 4, 5], [5, 6, 7]])
y = numpy.array([1, 0, 1, 1, 0, 1])
model = DecisionTreeClassifier()
model.fit(x, y)
# Sklearn model have X, how main argument for features
print("Predict directly from ml model")
print(model.predict(X=x))

up = mlup.UP(ml_model=model)
up.ml.load()
print("Predict from mlup.UP")
print(up.predict(X=x.tolist()))

# mlup.UP.web.load calling inner up.run_web_app method
up.run_web_app(daemon=True)
time.sleep(1)
resp = requests.post("http://0.0.0.0:8009/predict", json={"X": x.tolist()})

print("Predict from web app")
print(resp.json())
up.stop_web_app()
```

You can also check if the web application has been created and configured by using `loaded`.
```python
import mlup
from mlup.ml.empty import EmptyModel


up = mlup.UP(ml_model=EmptyModel())
print("Before mlup.UP.web.load()")
print(up.web.loaded)

up.ml.load()
print("After mlup.UP.ml.load()")
print(up.web.loaded)

up.web.load()
print("After mlup.UP.web.load()")
print(up.web.loaded)

up.conf.port = 8010
print("After mlup.UP.web.load() and change web app config")
print(up.web.loaded)
```

If you need to access a FastAPI-generated web application, you can do so through the `app` property.
```python
import mlup
from mlup.errors import WebAppLoadError
from mlup.ml.empty import EmptyModel
from fastapi import FastAPI

model = EmptyModel()
up = mlup.UP(ml_model=model)
up.ml.load()
print("Before mlup.UP.web.load()")
try:
    w = up.web.app
except WebAppLoadError:
    print("Web app not loaded for get created web app")

up.web.load()
print("After mlup.UP.web.load()")
print(up.web.app)
print(isinstance(up.web.app, FastAPI))
```
