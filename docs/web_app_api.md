# Web app API

The web application has an API of several methods:

* **[HEAD, OPTIONS, GET] /health** - Web application status. Must answer with code 200.
* **[GET]** /info - Information about the model, application, and its settings.
* **[POST]** /predict - call model prediction.
* **[GET]** /get-predict/{predict_id} - optional point. Obtaining prediction results.

## API

### /health

This is a simple web application state API method. Allows monitoring to ensure that the application is not frozen and can process requests.

#### HEAD

* status_code: 200

#### GET, OPTIONS

* status_code: 200
* body: `{"status_code": 200}`

### /info

This method returns information about the application and model, which allows the model client to know which version of the model and application it is working with, as well as some additional settings.

#### GET

* status_code: 200

If `debug=False` is specified in the application parameters (_by default `False`_), returns only informational data:
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
  },
}
```

If `debug=True` is specified in the application parameters (the default is `False`_), it returns the entire config for the model and for the web application:
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
    "max_thread_loop_workers": null,
    "data_transformer_for_predict": "mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer",
    "data_transformer_for_predicted": "mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer",
    "dtype_for_predict": null
  }
}
```

This allows you to debug and check what configuration is currently in use.

### /predict

The model's prediction itself. This is where the client sends data and receives the model's prediction.
The API method itself may change depending on the configuration.

#### POST

If the parameter `auto_detect_predict_params=False` (default is `True`_), then the method accepts data in 1 parameter `data_for_predict`.

**Request body**:
```json
{
  "data_for_predict": [
    // features_for_obj1: List,
    [1, 2, 3, 4, 5, 6],
    // features_for_obj2: List,
    [7, 6, 5, 4, 3, 2],
    // ...
  ]
}
```

If `auto_detect_predict_params=True` (default `True`_), then mlup analyzes the model and its predict method.
And converts the arguments of the predict method of the model into parameters accepted by this API method. (_See [Description of the application life cycle](https://github.com/nxexox/pymlup/tree/main/docs/life_cycle.md)_).

For example, for scikit-learn models, the `predict` method has 1 argument `X: Any`. And then the API request will look like this:

**Request body**:
```json
{
  "X": [
    // features_for_obj1: List,
    [1, 2, 3, 4, 5, 6],
    // features_for_obj2: List,
    [7, 6, 5, 4, 3, 2],
    // ...
  ]
}
```

If a model has several arguments, and they were all parsed, then they all end up in the API and can be passed into the model’s prediction method.

**Request body**:
```json
{
  "data_for_predict": [
    // features_for_obj1: List,
    [1, 2, 3, 4, 5, 6],
    // features_for_obj2: List,
    [7, 6, 5, 4, 3, 2],
    // ...
  ],
  "check_something": false,
  // ...
}
```

If no problems occurred, the method returns the prediction results.

* status_code: 200

```json
{
  "predict_result": [
    // predict result for obj1
    [1, 2, 3],
    // predict result for obj2
    [4, 2, 1]
    // ...
  ]
}
```
**In addition, the `X-Predict-id` header is always returned, even if an error occurs.**

The architecture may change the response body returned by this method. For example, for the architecture `mlup.web.architecture.worker_and_queue.WorkerAndQueueArchitecture` and `mlup.web.architecture.batching.BatchingSingleProcessArchitecture`,
if the `is_long_predict=True` configuration option is enabled, this method will return `predict_id`.
You can use it to pick up the results later. (_See [Web app architectures](https://github.com/nxexox/pymlup/tree/main/docs/web_app_architectures.md)_).

```json
{
  "predict_id": "7d9bea07-7505-45ec-a700-40417842025b"
}
```

### /get-predict/{predict_id}

This API method is only used in the `mlup.web.architecture.worker_and_queue.WorkerAndQueueArchitecture` and `mlup.web.architecture.batching.BatchingSingleProcessArchitecture` architectures.
If the model takes a long time to make a prediction, or the client wants to pick up the prediction results later, he can set the `is_long_predict=True` configuration parameter and go for the results later.
(_See [Web app architectures](https://github.com/nxexox/pymlup/tree/main/docs/web_app_architectures.md)_).

### GET

To get results, you need to send the required `predict_id` parameter to the URL.

If results for such a `predict_id` are found, they will be returned with status_code=200.
```json
// results for predict_id = "7d9bea07-7505-45ec-a700-40417842025b"
{
  "predict_result": [
    // predict result for obj1
    [1, 2, 3],
    // predict result for obj2
    [4, 2, 1]
    // ...
  ]
}
```

And if the prediction results were not found, then after `ttl_client_wait` has elapsed from the start of the request time, the application will return status_code 408.

## API Errors

All errors are in [errors](https://github.com/nxexox/pymlup/blob/main/mlup/errors.py).

All error APIs have a common response format.
**Response format:**
```json
{
  "detail": [
    {
      "loc": [
        "string"
      ],
      "msg": "string",
      "type": "string"
    }
  ],
  "predict_id": "string"
}
```

### 422

Validation error. If such an error is returned, it means that the sent data was not validated and the request was not accepted.

**Example answer:**
```json
{
  "detail": [
    {
      "loc": [
        "input_data",
        0
      ],
      "msg": "value is not a valid dict",
      "type": "type_error.dict"
    },
    {
      "loc": [
        "input_data",
        1,
        "col1"
      ],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ],
  "predict_id": "15eb8236-02bf-4658-9c04-cc53f37afdc3"
}
```

## 429

This error occurs when the request is not throttled.

**Example answer:**
```json
{
  "detail": [
    {
      "loc": [], 
      "msg": "Max requests in app. Please try again later.", 
      "type": "throttling_error"
    }
  ],
  "predict_id": "15eb8236-02bf-4658-9c04-cc53f37afdc3"
}
```

## 500

Error during model prediction or data transformation.

**Example answer:**
```json
{
  "detail": [
    {
      "loc": [],
      "msg": "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned while running Gather node. Name:'Gather_2' Status Message: indices element out of data bounds, idx=123124124124 must be within the inclusive range [-250002,250001]",
      "type": "predict_error"
    }
  ],
  "predict_id": "0c4b7afe-032e-442f-b265-7773d9d78580"
}
```

## Свой API

As shown in the [life_cycle.md](https://github.com/nxexox/pymlup/tree/main/docs/life_cycle.md#web-application-customization) section, you can completely customize a web application after it has been created.

Including, you can add new or even change existing API methods.