# Config file

The mlup application configuration consists of two blocks: `ml`, `web`.
When you configure using Python code, this separation does not exist. It is only in the config files.

You don't always have to write a complete configuration file. You can explicitly specify only those parameters that differ from the default ones.
All other parameters will be filled with default values.

## What settings are there?

### ml

#### Information arguments
`name (str)` - Model name. Use for information about your model.

Default is "MyFirstMLupModel".

---

`version (str)` - Model version. Use for information about your model. 

Default is "1.0.0.0".

---

#### Model interface settings
`type (ModelLibraryType)` - Model type for choose current code for load and run model.
    See [mlup.constants.ModelLibraryType](https://github.com/nxexox/pymlup/blob/main/mlup/constants.py) for details.
    
Default is `ModelLibraryType.SKLEARN`.

---

`columns (Optional[List[Dict[str, str]]])` - Columns description for model predict method.
    Format: List[Dict].

    Example [{"name": "col1", "type": "str", "required": True, "default": None, "collection_type": "List"}]
    name - Column name;
    type - Column type in string: int, str, float, bool.
    required - bool and optional field. By default is True.
    default - Any data and optional.
    collection_type - Type of collection is optional field. Supported: List. Default is None.
    
If you not set columns, then columns validation should be False. 

---

`predict_method_name (str)` - Name for predict method by your model.
    This param use for analyze predict method and run predict method in your model.
    
Default is "predict".

---

#### Model load settings
`auto_detect_predict_params (bool)` - If set is True, UP will parse predict method your model
    and create from predict method arguments, validators and ApiScheme for inner data.
    If set is False, then UP wait json with {data_for_predict: [..data for predict..]}.
    
Default is True.

---

`storage_type (StorageType)` - Type storage, where should the model be loaded from.
    All storages code there is in [mlup.ml.storage](https://github.com/nxexox/pymlup/blob/main/mlup/ml/storage/) package.
    This param use in load() method, by it search storage class and use for load model.
    Params for storage class, can set to storage_kwargs.
    If set `StorageType.memory`, you can set ml to construct, without load from storage.
    
You can write your storage class, for load model from you storage or with your logic. 
For it, your storage class must be a class [mlup.ml.storage.base.BaseStorage](https://github.com/nxexox/pymlup/blob/main/mlup/ml/storage/base.py) heir.
See [Storages](storages.md) for details.

Default: StorageType.memory.

---

`storage_kwargs (Dict)` - Naming arguments for storage class from storage_type.

Default: empty Dict.

---

`binarization_type (BinarizationType)` - Type binarization model method.
    Use for load binary model from storage. See [Binarizers](binarizers.md) for details.
    
If set "auto", run auto search binarization type by model raw binary data and model name in storage.
If binarization_type="auto" found binarization, than set binarization_type to found binarizer class.
If binarization_type="auto" not found binarization, than raise [mlup.errors.ModelLoadError](https://github.com/nxexox/pymlup/blob/main/mlup/errors.py).

Default is "auto".

---

#### Model work settings
`use_thread_loop (bool)` - Use concurrent.futures.ThreadPoolExecutor for create workers
    and run in workers pool predict in model.
    This useful for not blocking App CPU bound operations on the predict time.

Default is True.

---

`max_thread_loop_workers (Optional[int])` - Count thread workers in `concurrent.futures.ThreadPoolExecutor`.
    Param use if `use_thread_loop` is set True. If not set this param, then will be calculate,
    how calculate concurrent library - min(32, os.cpu_count() + 4).

Default is None, that is min(32, os.cpu_count() + 4).

---

`data_transformer_for_predict (ModelDataTransformerType)` - How data type need for model predict method.
    See [Data Transformers](data_transformers.md) for details.
    
Default ModelDataTransformerType.NUMPY_ARR.

---

`data_transformer_for_predicted (ModelDataTransformerType)` - How data type model returned from predict method.
    See [Data Transformers](data_transformers.md) for details.

Default ModelDataTransformerType.NUMPY_ARR.

---

`dtype_for_predict (Optional[str])` - Dtype for data_transformer_for_predict.
    Each data_transformer uses its own way of searching for a dtype by the specified one.
    But as a rule it is getattr([pandas, numpy, torch, tensorflow], dtype).
See [Data Transformers](data_transformers.md) for details.

Default is None, and the library itself determines which type.

### web

#### WebApp web outer interface settings

`host (str)` - Web host for web app.
    
Default is "0.0.0.0".

---

`port (int)` - Web port for web app.
    
Default is 8009.

---

`web_app_version (str)` - User version for web_app.
    Using the mlup version may not be nice to users your web app.
    However, you can change the library version without changing the model.
    And then the code of the web application backend actually changes.
    You can use this field to notify users of these changes.

Default is "1.0.0.0".

---

`column_validation (bool)` - Do I need to create a validator for columns from columns config
    that are sent by the user through the web application?
    
Default is False.

---

`custom_column_pydantic_model (Optional[Type[PydanticBaseModel]])` - Custom column pydantic model.
    Use for validation. If set this field, flag column_validation not use. Validation will be always.

Default is None.

---

#### WebApp architecture settings

`mode (WebAppArchitecture)` - Work mode web app. We have three modes for work:
    directly_to_predict, worker_and_queue, batching. 
    See [Web app architectures](web_app_architectures.md) for details.
    
Default is WebAppArchitecture.directly_to_predict.

---

`max_queue_size (int)` - Max queue size with inner data for model.
    It's use in modes worker_and_queue and batching, for controlling queue inner requests size.

Default is 100.

---

`ttl_predicted_data (int)` - Max time member predict result data.
    It's need for clean memory, if client not returned for predict result.

Default is 60.

---

`ttl_client_wait (float)` - Max time wait results for clients, in single request.
    If the client has not waited for a response within this time, the server responds with a 408 response code, and the user must make a second request.
    This is necessary so as not to make customers wait forever and all the ensuing consequences.

Default is 30.0

---

`min_batch_len (int)` - Min batch len for start predict.
    Batching is regulated by two parameters: the batch accumulation time and the size of the accumulated batch.
    If one of the parameters has reached its limit, then the batch predictor is launched.
    It's min batch size for call batch predict.

Default is 10.

---

`batch_worker_timeout (float)` - Max time for pending before run batching.
    Batching is regulated by two parameters: the batch accumulation time and the size of the accumulated batch.
    If one of the parameters has reached its limit, then the batch predictor is launched.
    It's max time waiting batch for call batch predict.

Default is 1.0.

---

`is_long_predict (bool)` - Added get-predict api method and add return predict_id to predict response.
    Use only in worker_and_queue and batching modes.
    If set this param, after request to /predict, client get predict_id in response.
    If not set this param, then after `ttl_client_wait` time, client can't get predict result.

Default is False.

---

#### WebApp work settings

`show_docs (bool)` - Enable API docs in web app by URL /docs.

Default is True

---

`debug (bool)` - Debug mode. Use for FastAPI and for show model configs.

Default is False

---

`throttling_max_requests (Optional[int])` - Max count simultaneous requests to web app.
    If set None, throttling by requests is disabling.

Default is None

---

`throttling_max_request_len (Optional[int])` - Max count objects to predict in single request.
    If set None, throttling by max items in request is disable.

Default is None

---

`timeout_for_shutdown_daemon (float)`- Wait time for graceful shutdown web app.

Default is 3.0

---

`uvicorn_kwargs (Dict)` - Uvicorn server kwargs arguments.

Default is {}

---

`item_id_col_name (str)` - Column name for unique item_id.
    Use in batching and worker. It need for marks requests in worker by request id.
    Added this key for predicted data, but not thrown into model predict.
    
Default is [mlup.constants.ITEM_ID_COL_NAME](https://github.com/nxexox/pymlup/blob/main/mlup/constants.py) - "mlup_item_id".

---

## Default configuration file

### yaml

```yaml
version: '1'
ml:
  auto_detect_predict_params: true
  binarization_type: auto
  columns: null
  data_transformer_for_predict: mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer
  data_transformer_for_predicted: mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer
  dtype_for_predict: null
  max_thread_loop_workers: null
  name: MyFirstMLupModel
  predict_method_name: predict
  storage_type: mlup.ml.storage.memory.MemoryStorage
  type: sklearn
  use_thread_loop: true
  version: 1.0.0.0
web:
  batch_worker_timeout: 1.0
  column_validation: false
  debug: false
  host: 0.0.0.0
  is_long_predict: false
  item_id_col_name: mlup_item_id
  max_queue_size: 100
  min_batch_len: 10
  mode: mlup.web.architecture.directly_to_predict.DirectlyToPredictArchitecture
  port: 8009
  show_docs: true
  throttling_max_request_len: null
  throttling_max_requests: null
  timeout_for_shutdown_daemon: 3.0
  ttl_client_wait: 30.0
  ttl_predicted_data: 60
  uvicorn_kwargs: {}
  web_app_version: 1.0.0.0
```

### json

```json
{
  "version": "1",
  "ml": {
    "storage_type": "mlup.ml.storage.memory.MemoryStorage", 
    "data_transformer_for_predict": "mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer", 
    "type": "sklearn", 
    "data_transformer_for_predicted": "mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer", 
    "auto_detect_predict_params": true, 
    "version": "1.0.0.0", 
    "binarization_type": "auto", 
    "max_thread_loop_workers": null, 
    "dtype_for_predict": null, 
    "use_thread_loop": true, 
    "predict_method_name": "predict", 
    "columns": null, 
    "name": "MyFirstMLupModel"
  }, 
  "web": {
    "port": 8009, 
    "show_docs": true, 
    "ttl_client_wait": 30.0, 
    "timeout_for_shutdown_daemon": 3.0, 
    "uvicorn_kwargs": {}, 
    "batch_worker_timeout": 1.0, 
    "item_id_col_name": "mlup_item_id", 
    "throttling_max_request_len": null, 
    "is_long_predict": false, 
    "max_queue_size": 100, 
    "host": "0.0.0.0", 
    "min_batch_len": 10, 
    "debug": false, 
    "web_app_version": "1.0.0.0", 
    "mode": "mlup.web.architecture.directly_to_predict.DirectlyToPredictArchitecture", 
    "throttling_max_requests": null, 
    "column_validation": false, 
    "ttl_predicted_data": 60
  }
}
```

## How to get the default configuration file

You can get the default configuration file in several ways, for example:
```python
import mlup
from mlup.ml.empty import EmptyModel

mlup.generate_default_config('/path/to/your/config/file.yaml')
mlup.generate_default_config('/path/to/your/config/file.json')
up = mlup.UP(ml_model=EmptyModel())
up.to_yaml('/path/to/your/config/file.yaml')
up.to_json('/path/to/your/config/file.json')
dict_config = up.to_dict()
```

## Config baselines

In addition to the default config, mlup created configs for basic scenarios, changing the settings in them for a specific scenario.
For example, for a tensorflow model or for a web app in batching mode.

```python
from dataclasses import dataclass
from typing import Union, Optional

from mlup.constants import ModelLibraryType, ModelDataTransformerType, BinarizationType, WebAppArchitecture
from mlup.up import Config


@dataclass
class WorkerAndQueueConfig(Config):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class BatchingConfig(Config):
    mode: WebAppArchitecture = WebAppArchitecture.batching


@dataclass
class ScikitLearnConfig(Config):
    type: Union[str, ModelLibraryType] = ModelLibraryType.SCIKIT_LEARN
    predict_method_name: str = 'predict'


@dataclass
class ScikitLearnWorkerConfig(ScikitLearnConfig):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class ScikitLearnBatchingConfig(ScikitLearnConfig):
    mode: WebAppArchitecture = WebAppArchitecture.batching


@dataclass
class TensorflowConfig(Config):
    type: Union[str, ModelLibraryType] = ModelLibraryType.TENSORFLOW
    predict_method_name: str = '__call__'
    data_transformer_for_predict: Union[str, ModelDataTransformerType] = ModelDataTransformerType.TENSORFLOW_TENSOR
    data_transformer_for_predicted: Union[str, ModelDataTransformerType] = ModelDataTransformerType.TENSORFLOW_TENSOR
    dtype_for_predict: Optional[str] = 'float32'


@dataclass
class TensorflowWorkerConfig(TensorflowConfig):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class TensorflowBatchingConfig(TensorflowConfig):
    mode: WebAppArchitecture = WebAppArchitecture.batching


@dataclass
class TorchConfig(Config):
    type: Union[str, ModelLibraryType] = ModelLibraryType.TORCH
    predict_method_name: str = '__call__'
    data_transformer_for_predict: Union[str, ModelDataTransformerType] = ModelDataTransformerType.TORCH_TENSOR
    data_transformer_for_predicted: Union[str, ModelDataTransformerType] = ModelDataTransformerType.TORCH_TENSOR
    dtype_for_predict: Optional[str] = 'float32'


@dataclass
class TorchWorkerConfig(TorchConfig):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class TorchBatchingConfig(TorchConfig):
    mode: WebAppArchitecture = WebAppArchitecture.batching


@dataclass
class OnnxConfig(Config):
    type: Union[str, ModelLibraryType] = ModelLibraryType.ONNX
    binarization_type: Union[str, BinarizationType] = BinarizationType.ONNX_INFERENCE_SESSION


@dataclass
class OnnxWorkerConfig(OnnxConfig):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class OnnxBatchingConfig(OnnxConfig):
    mode: WebAppArchitecture = WebAppArchitecture.batching
```

Using this config instead of the standard one is quite simple:

```python
import mlup
from mlup import baseline
from mlup.ml.empty import EmptyModel

up = mlup.UP(
    ml_model=EmptyModel(), 
    conf=baseline.BatchingConfig(),
)
```
