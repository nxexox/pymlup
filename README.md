# PyMLup

It's library for easy run ML in production. 

All you need is to deliver the model file and config to the server (in fact, the config is not necessary) 🙃

PyMLup is a modern way to run machine learning models in production. The market time has been reduced to a minimum. This library eliminates the need to write your own web applications with machine learning models and copy application code. It is enough to have a machine learning model to launch a web application with one command.

* It's library learning only clean python;
* Use FastApi in web app backend;

Work tested with machine learning model frameworks (links to tests):
* [scikit-learn>=1.2.0,<1.3.0](tests/integration_tests/frameworks/test_scikit_learn_model.py)
* [tensorflow>=2.0.0,<3.0.0](tests/integration_tests/frameworks/test_tensorflow_model.py)
* [lightgbm>=4.0.0,<5.0.0](tests/integration_tests/frameworks/test_lightgbm_model.py)
* [torch>=2.0.0,<3.0.0](tests/integration_tests/frameworks/test_pytorch_model.py)
* [onnx>=1.0.0,<2.0.0](tests/unit_tests/ml/test_binarization.py)
* [onnxruntime>=1.0.0,<2.0.0](tests/unit_tests/ml/test_binarization.py)

Support and tested with machine learning libraries:
* [numpy>=1.0.0,<2.0.0](tests/unit_tests/ml/test_data_transformers.py)
* [pandas>=2.0.0,<3.0.0](tests/unit_tests/ml/test_data_transformers.py)
* [joblib>=1.2.0,<1.3.0](tests/unit_tests/ml/test_binarization.py)
* [tf2onnx>=1.0.0,<2.0.0](tests/unit_tests/ml/test_binarization.py)
* [skl2onnx>=1.0.0,<2.0.0](tests/unit_tests/ml/test_binarization.py)
* [jupyter==1.0.0](tests/integration_tests/test_jupyter_notebook.py)

**The easiest way to poke:**
```bash
pip install pymlup
mlup run -m /path/to/my/model.onnx
```

## Useful links
* [Docs](docs) under development;
* [Examples](examples);
* [Tests models](mldata)

## How it's work

1. You are making your machine learning model. Optional: you are making mlup config for your model.
2. You deliver your model to server. Optional: you deliver your config to server.
3. Installing pymlup to your server and libraries for model.
4. Run web app from your model or your config 🙃

## Requirements

Python 3.7+

* PyMLup stands on the shoulders of giants FastAPI for the web parts. 
* Additionally, you need to install the libraries that your model uses.

## Installation

```bash
pip install pymlup
```

You will also can install with ml backend library:
```bash
pip install "pymlup[scikit-learn]"  # For scikit-learn
pip install "pymlup[lightgbm]"      # For microsoft lightgbm
pip install "pymlup[tensorflow]"    # For tensorflow
pip install "pymlup[torch]"         # For torch
pip install "pymlup[onnx]"          # For onnx models: torch, tensorflow, sklearn, etc...
```

## Examples

### Examples code

```python
import mlup

class MyAnyModelForExample:
    def predict(self, X):
        return X

ml_model = MyAnyModelForExample()


up = mlup.UP(ml_model=ml_model)
# Need call up.ml.load(), for analyze your model
up.ml.load()
# If you want testing your web app, you can run in daemon mode
# You can open browser http://localhost:8009/docs
up.run_web_app(daemon=True)

import requests
response = requests.post('http://0.0.0.0:8009/predict', json={'X': [[1, 2, 3], [4, 5, 6]]})
print(response.json())

up.stop_web_app()
```

You can check work model by config, without web application.
* `predict` - Get model predict as inner arguments as in web app.
* `predict_from` - As `predict` method, but not use data transformer before call model predict.
* `async_predict` - Asynchronous version of the `predict` method.
```python
import mlup
import numpy

class MyAnyModelForExample:
    def predict(self, X):
        return X

ml_model = MyAnyModelForExample()
up = mlup.UP(ml_model=ml_model)
up.ml.load()

up.predict(X=[[1, 2, 3], [4, 5, 6]])
up.predict_from(X=numpy.array([[1, 2, 3], [4, 5, 6]]))
await up.async_predict(X=[[1, 2, 3], [4, 5, 6]])
```

#### Save ready application to disk

##### Make default config

If path endswith to json, make json config, else yaml config.

```python
import mlup
mlup.generate_default_config('path_to_yaml_config.yaml')
```

##### From config

You can save ready config to disk, but you need set local storage and path to model file in server.
In folder can there are many files, mask need for filter exactly our model file

```python
import mlup
from mlup.ml.empty import EmptyModel  # This stub class
from mlup import constants

up = mlup.UP(ml_model=EmptyModel())
up.conf.storage_type = constants.StorageType.disk
up.conf.storage_kwargs = {
    'path_to_files': 'path/to/model/file/in/model_name.modelextension',
    'file_mask': 'model_name.modelextension',
}
up.to_yaml("path_to_yaml_config.yaml")
up.to_json("path_to_json_config.json")

# After in server
up = mlup.UP.load_from_yaml("path_to_yaml_config.yaml", load_model=True)
up.run_web_app()
```

##### From pickle

If you make pickle/joblib file your mlup with model, don't need to change storage type, because your model there is in your pickle/joblib file.

```python
import pickle
import mlup
from mlup.ml.empty import EmptyModel  # This stub class

up = mlup.UP(ml_model=EmptyModel())

# You can create pickle file
with open('path_to_pickle_file.pckl', 'wb') as f:
    pickle.dump(up, f)

# After in server
with open('path_to_pickle_file.pckl', 'rb') as f:
    up = pickle.load(f)
up.ml.load()
up.run_web_app()
```

#### Change config

If you can change model settings (See config attributes docs), need call `up.ml.load_model_settings()`.

```python
import mlup

class MyAnyModelForExample:
    def predict(self, X):
        return X

ml_model = MyAnyModelForExample()

up = mlup.UP(
    ml_model=ml_model,
    conf=mlup.Config(port=8011)
)
up.ml.load()
up.conf.auto_detect_predict_params = False
up.ml.load_model_settings()
```

### Examples server commands

#### mlup run

You can run web application from model, config, pickle up object. Bash command mlup run making this.

See `mlup run --help` for full docs.

##### From model
```bash
mlup run -m /path/to/your/model.extension
```

This will run code something like this: 

```python
import mlup
from mlup import constants

up = mlup.UP(
    conf=mlup.Config(
        storage_type=constants.StorageType.disk,
        storage_kwargs={
            'path_to_files': '/path/to/your/model.extension',
            'files_mask': r'.+',
        },
    )
)
up.ml.load()
up.run_web_app()
```

You change config attributes in this mode. For this, you can add arguments `--up.<config_attribute_name>=new_value`. 
(For more examples see `mlup run --help`).

##### From config
```bash
mlup run -c /path/to/your/config.yaml
# or mlup run -ct json -c /path/to/your/config.json
```

This will run code something like this:

```python
import mlup

up = mlup.UP.load_from_yaml(conf_path='/path/to/your/config.yaml', load_model=True)
up.run_web_app()
```

##### From mlup.UP pickle/joblib object
```bash
mlup run -b /path/to/your/up_object.pckl
# or mlup run -bt joblib -b /path/to/your/up_object.joblib
```

This will run code something like this:

```python
import pickle

with open('/path/to/your/up_object.pckl', 'rb') as f:
    up = pickle.load(f)
up.run_web_app()
```

#### mlup make-app

This command making `.py` file with mlup web application and your model, config, pickle up object or with default settings.

See `mlup make-app --help` for full docs.

##### With default settings
```bash
mlup make-app example_without_data_app.py
```

This command is making something like this:

```python
# example_without_data_app.py
import mlup


# You can load the model yourself and pass it to the "ml_model" argument.
# up = mlup.UP(ml_model=my_model, conf=mlup.Config())
up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        # You can use storage_type and storage_kwargs for auto_load model from storage.
    )
)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn example_app:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
```

And you can write your settings and run web application:
```bash
python3 example_without_data_app.py
```

##### With only model
```bash
mlup make-app -ms /path/to/my/model.onnx example_without_data_app.py
```

This command is making something like this:

```python
# example_without_data_app.py
import mlup
from mlup import constants


up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        storage_type=constants.StorageType.disk,
        storage_kwargs={
            'path_to_files': '/path/to/my/model.onnx',
            'files_mask': 'model.onnx',
        },
    )
)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn example_app:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()

```

And you can run web application:
```bash
python3 example_without_data_app.py
```

##### With only config
```bash
mlup make-app -cs /path/to/my/config.yaml example_without_data_app.py
```

This command is making something like this:

```python
# example_without_data_app.py
import mlup


up = mlup.UP.load_from_yaml('/path/to/my/config.yaml', load_model=False)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn example_app:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
```

And you can run web application:
```bash
python3 example_without_data_app.py
```

##### With only binary UP object
```bash
mlup make-app -bs /path/to/my/up.pickle example_without_data_app.py
```

This command is making something like this:

```python
# example_without_data_app.py
import pickle


with open('/path/to/my/up.pickle', 'rb') as f:
    up = pickle.load(f)

if not up.ml.loaded:
    up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn example_app:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
```

And you can run web application:
```bash
python3 example_without_data_app.py
```

#### mlup validate-config

This command use for validation your config. This command have alpha version and need finalize.

See `mlup validate-config --help` for full docs.

```bash
mlup validate-config /path/to/my/conf.yaml
```

## Web application interface

By default, web application starting on http://localhost:8009 and have api docs.

### Interactive API docs

Now go to http://localhost:8009/docs.

You will see the automatic interactive API documentation (provided by [Swagger UI](https://github.com/swagger-api/swagger-ui)):

### Api points

#### /health
Use for check health web application.
 
HTTP's methods: HEAD, OPTIONS, GET

<details>

##### Return JSON 
```{'status': 200}``` and status code is 200.

</details>

#### /info
Use for get model and application information. If set debug=True in config, return full config. 

HTTP's methods: GET

<details>

##### Return JSON:
```json
{
  "model_info": {
    "name": "MyFirstMLupModel",
    "version": "1.0.0.0",
    "type": "sklearn",
    "columns": null
  },
  "web_app_info": {
    "version": "1.0.0.0",
    "column_validation": false,
    "debug": false
  }
}
```

If set in config `debug=True`, return another json, almost complete config. But no sensitive data.

```json
{
  "web_app_config": {
    "host": "localhost",
    "port": 8009,
    "web_app_version": "1.0.0.0",
    "column_validation": false,
    "custom_column_pydantic_model": null,
    "mode": "mlup.web.architecture.directly_to_predict.DirectlyToPredictArchitecture",
    "max_queue_size": 100,
    "ttl_predicted_data": 60,
    "ttl_client_wait": 30.0,
    "min_batch_len": 10,
    "batch_worker_timeout": 1.0,
    "is_long_predict": false,
    "show_docs": true,
    "debug": true,
    "throttling_max_requests": null,
    "throttling_max_request_len": null,
    "timeout_for_shutdown_daemon": 3.0,
    "item_id_col_name": "mlup_item_id"
  },
  "model_config": {
    "name": "MyFirstMLupModel",
    "version": "1.0.0.0",
    "type": "sklearn",
    "columns": null,
    "predict_method_name": "predict",
    "auto_detect_predict_params": true,
    "storage_type": "mlup.ml.storage.memory.MemoryStorage",
    "binarization_type": "auto",
    "use_thread_loop": true,
    "max_thread_loop_workers": true,
    "data_transformer_for_predict": "mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer",
    "data_transformer_for_predicted": "mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer",
    "dtype_for_predict": null
  }
}
```

</details>

#### /predict

Use for call predict in model.

HTTP's methods: POST

<details>

##### Requests body data:
```json
{
  "data_for_predict": [
    "input_data_for_obj_1",
    "input_data_for_obj_2",
    "input_data_for_obj_3"
  ]
}
```

Key `data_for_predict` is default key for inner data. In config by default set param `auto_detect_predict_params` is True. 
This param activate analyze model predict method, get arguments from and generate API by params. 
If `auto_detect_predict_params` found params, he changes `data_for_predict` to finding keys and change API docs.

Example for `scikit-learn` models:
```json
{
  "X": [
    "input_data_for_obj_1",
    "input_data_for_obj_2",
    "input_data_for_obj_3"
  ]
}
```

`input_data_for_obj_1` maybe any valid JSON data. These data are run through data transformers from config `data_transformer_for_predict`.

By default, this param is `mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer`.

##### Return JSON:
```json
{
  "predict_result": [
    "predict_result_for_obj_1",
    "predict_result_for_obj_2",
    "predict_result_for_obj_3"
  ]
}
```

`predict_result_for_obj_1` will be valid JSON data. These data, after being predicted by the model, are run through data transformers from config `data_transformer_for_predicted`.

By default, this param is `mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer`.

</details>


##### Validation

This method have validation for inner request data. It's making from config `columns` and flag `column_validation`.

## Web application modes

Web application have three works modes:
* `directly_to_predict` - is Default. User request send directly to model.
* `worker_and_queue` - ml model starts in thread worker and take data for predict from queue. 
  Web application new user requests send to queue and wait result from results queue.
* `batching` - ml model start in thread worker and take data for predict from queue. 
  But not for one request, but combines data from several requests and sends it in one large array to the model. 
  Web application new user requests send to queue and wait result from results queue.

This param is naming `mode`.
```python
import mlup
from mlup.ml.empty import EmptyModel
from mlup import constants

up = mlup.UP(
    ml_model=EmptyModel(),
    conf=mlup.Config(
        mode=constants.WebAppArchitecture.worker_and_queue,
    )
)
```

If your model is light, or you hae many CPU/GPU/RAM, you can run many processes:
```python
import mlup
from mlup.ml.empty import EmptyModel
from mlup import constants

up = mlup.UP(
    ml_model=EmptyModel(),
    conf=mlup.Config(
        mode=constants.WebAppArchitecture.worker_and_queue,
        uvicorn_kwargs={'workers': 4},
    )
)
```

## TODO

* Full docs by all pymlup library.
* Added length inner array validation.
* Modern mlup validate-config command.
* Auto search params by model and test data.
* Support multi models pipelines. Now support is only single file model and single model predict method.
* Add C++ implementation this application by config or model. Support everything that is possible according to the mlup config.
* Add hub for save configs.
