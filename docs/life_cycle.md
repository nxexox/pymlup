# Life cycle

The entire life process of an mlup application can be divided into two stages - application initialization and running application.

## Initializing the application

mlup consists of two main components - [ml](https://github.com/nxexox/pymlup/tree/main/mlup/ml) and [web](https://github.com/nxexox/pymlup/tree/main/mlup/web).

### mlup.ml

This component contains all the code for working with your model, including all settings associated with the model.

When you create a `mlup.UP` object and pass a model `mlup.UP(ml_model=your_model)` into it, mlup does nothing.
At this point, you have simply created a class. This is done in order to avoid starting the processes of loading and analyzing the model, which can be lengthy, without explicit instructions.

TODO: HERE IS A PICTURE DIAGRAM WITH METHODS AND APPLICATION LIFE CYCLE

#### up.ml.load()

To start using your model, you need to load the model into mlup - `up.ml.load()`.
At this point, mlup first loads your model into memory. In the case of `mlup.UP(ml_model=your_model)` your model is already loaded and mlup will not load it into memory again.

Loading a model into memory consists of two stages:
* Loading binary data using `storage_type`;
* Deserilization of binary data into the model;

If you use `storage_type=mlup.constants.StorageType.disk`, by default binary data is not loaded into memory and binarizers themselves load the model during binarization.
This was done to eliminate the possibility of duplicating model data in memory. Binaryizers have access to a local disk, so it is not difficult for them to load the model themselves.
To change this behavior, `storage_type=mlup.constants.StorageType.disk` has a `need_load_file` flag, which defaults to False.

The storage concept is needed to load binary model data from different storages to a local disk, from where binarizers will read the data.

Once the model is loaded in memory, `up.ml.load_model_settings()` is called inside `up.ml.load()`.

You can check whether the model has been loaded into memory using the `up.ml.loaded: bool` property. It becomes True only when the model is in memory. 
But it does not indicate whether the model was analyzed using the `up.ml.load_model_settings()` method.

#### up.ml.load_model_settings()

The `up.ml.load_model_settings()` method analyzes the loaded model, according to the settings specified in the config, and prepares the `mlup.UP` object to work with it.

At this point the analysis occurs:
* Method for prediction;
* Method arguments for prediction;
* Creating data transformers to convert user data into model format and model response into user format;
* Creation of auxiliary entities, such as `concurrent.futures.ThreadPoolExecutor`;

After this, your model is ready to be used via mlup.

If you change some setting related to the operation of the model, just call `up.load_model_settings()`.
Then mlup will not reload your model into memory, but will simply analyze it again taking into account the changed config.

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
up.predict_from(X=numpy.array(objs_for_predict))

up.conf.auto_detect_predict_params = False
# Refresh mlup model settings
up.ml.load_model_settings()

up.predict_from(data_for_predict=numpy.array(objs_for_predict))
```

### mlup.web

To launch a web application, it also needs to be initialized with `up.web.load()`.

When using `up.run_web_app()`, you don't need to worry about this - mlup will initialize it itself.
But if the web component has already been initialized, you need to specify the `up.run_web_app(force_load=True)` parameter.
If you want to launch your web application in any of the other ways, you need to take care of initializing it yourself - call the `up.web.load()` method.

_P.S. You don't have to worry if you forget to call `up.web.load()`. mlup will not be able to launch the web application and will crash with the error [WebAppLoadError](https://github.com/nxexox/pymlup/blob/main/mlup/errors.py)_

Unlike the ml component, the web component has only 1 loading method, `up.web.load()`, which each time recreates the web app application and reinitializes all settings.

Because the web component builds the API and validation for incoming data, it uses the results of your model analysis.
This means that you will not be able to initialize web first and then ml. The web component needs an initialized ml component to initialize.

Just like in the case of ml, web has the `up.web.loaded` attribute. It becomes True if and only if a `fastapi.FastAPI` application is created. 
Those only after calling `up.web.load()`.

If your application has already called `up.web.load()`, and then you change the settings, this attribute will still be True. 
You need to independently monitor the reinitialization of the application after updating the config.

```python
import mlup

class MyModel:
    def predict(self, X):
        return X

model = MyModel()

up = mlup.UP(ml_model=model)
up.ml.load()

obj_1 = [1, 2, 3]
obj_2 = [4, 5, 6]
objs_for_predict = [obj_1, obj_2]
model_predicted = model.predict(X=objs_for_predict)
mlup_predicted = up.predict(X=objs_for_predict)

print(f'Before call up.web.load: {up.web.loaded}')
up.web.load()
print(f'After call up.web.load and before change config: {up.web.loaded}')
up.conf.port = 8011
print(f'After call up.web.load and after change config: {up.web.loaded}')
up.web.load()
print(f'After double call up.web.load: {up.web.loaded}')
up.run_web_app()

import requests
resp = requests.post('http://0.0.0.0:8011/predict', json={'X': objs_for_predict})
web_app_predicted = resp.json()

up.stop_web_app()

print(model_predicted)
print(mlup_predicted)
print(web_app_predicted)
```

After successful initialization of the web component, you can launch your application in a way convenient for you.

## Web application customization

If you need to add your own code to initialize the FastAPI application, it is available through the `up.web.app: fastapi.FastAPI` attribute.

The web application becomes available only after calling `up.web.load()`. In this method it's created.

```python
import mlup
from mlup.ml.empty import EmptyModel


up = mlup.UP(ml_model=EmptyModel(), conf=mlup.Config())
up.ml.load()
up.web.load()

app = up.web.app


def my_api_method():
    return {}


app.add_api_route("/my-api-method", my_api_method, methods=["GET"], name="my-api-method")
up.run_web_app()
```

