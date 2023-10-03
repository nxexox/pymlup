# Data transformers

As described in [Description of the application life cycle](https://github.com/nxexox/pymlup/tree/main/docs/life_cycle.md#ml-predict-process), data transformers are needed to convert data from JSON format to model format and back.
For example, from python list to numpy.array and back.

mlup comes with several data transformers out of the box. This set corresponds to the supported binarization methods.

* [mlup.ml.data_transformers.pandas_data_transformer.PandasDataFrameTransformer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/data_transformers/pandas_data_transformer.py) - docs is [here](https://pandas.pydata.org/).
* [mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/data_transformers/numpy_data_transformer.py) - docs is [here](https://numpy.org/).
* [mlup.ml.data_transformers.tf_tensor_data_transformer.TFTensorDataTransformer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/data_transformers/tf_tensor_data_transformer.py) - docs is [here](https://www.tensorflow.org/api_docs/python/tf/Tensor).
* [mlup.ml.data_transformers.torch_tensor_data_transformer.TorchTensorDataTransformer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/data_transformers/torch_tensor_data_transformer.py) - docs is [here](https://pytorch.org/docs/stable/tensors.html).
* [mlup.ml.data_transformers.src_data_transformer.SrcDataTransformer](https://github.com/nxexox/pymlup/blob/main/mlup/ml/data_transformers/src_data_transformer.py) - docs in this document.

These transformers can transform data from JSON format into their own format and back. There are two methods for this: `transform_to_model_format` and `transform_to_json_format`.

## mlup transformers

You can specify data transformers using the `data_transformer_for_predict` and `data_transformer_for_predicted` configuration parameters.
Which are used to convert the data into model format and convert the response from the model accordingly.

### mlup.ml.data_transformers.pandas_data_transformer.PandasDataFrameTransformer

This transformer converts the incoming query into pandas tabular data. Since tabular data has column names, the column names and types are the data interface to the model's prediction.
It is necessary to specify the columns, either in the configuration parameter [columns](https://github.com/nxexox/pymlup/tree/main/docs/config_file.md#model-interface-settings) or send a dictionary with columns directly in the request.

```python
import mlup
from mlup import constants

class Model:
    def predict(self, X):
        return X

columns = [
    {"name": "col1", "type": "int"},
    {"name": "col2", "type": "str"},
    {"name": "col3", "type": "str"},
]
obj1 = [1, 2, 3]
obj2 = [4, 5, 6]
data_without_columns = [obj1, obj2]
data_with_columns = [
    {c["name"]: v for c, v in zip(columns, obj1)}, 
    {c["name"]: v for c, v in zip(columns, obj2)}
]

# With columns in config
up = mlup.UP(
    ml_model=Model(),
    conf=mlup.Config(
        columns=columns,
        data_transformer_for_predict=constants.ModelDataTransformerType.PANDAS_DF,
        data_transformer_for_predicted=constants.ModelDataTransformerType.PANDAS_DF,
    )
)
up.ml.load()
print(up.predict(X=data_without_columns))
# [{'col1': 1, 'col2': 2, 'col3': 3}, {'col1': 4, 'col2': 5, 'col3': 6}]

up.conf.columns = None
up.ml.load(force_loading=True)

print(up.predict(X=data_with_columns))
# [{'col1': 1, 'col2': 2, 'col3': 3}, {'col1': 4, 'col2': 5, 'col3': 6}]
```

For this data transformer, when creating pandas.DataFrame, you can specify dtype. To do this, use the `dtype_for_predict` configuration parameter.
In this parameter, you can specify the Dtype name string in pandas. Here are some of them: `Float32Dtype`, `Float64Dtype`, `Int8Dtype`, `Int16Dtype`, `Int32Dtype`, `StringDtype`, `BooleanDtype`.
If you do not specify the `dtype_for_predict` parameter in the configuration, then pandas.DataFrame determines the types itself.

To reverse the conversion, call `pandas.DataFrame(...).to_dict("records")`.

### mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer

This data transformer is used by default to transform data to and from model format.

Regardless of the presence of columns in the query or [config columns](https://github.com/nxexox/pymlup/tree/main/docs/config_file.md#model-interface-settings), after this transformer you will get `numpy.array`.
If you have columns specified in the configuration or the data is sent with columns, they will affect the order of the resulting array.
The data in the final array will be in the same order in which the columns are specified in the config or serialized by the web framework and turned into List[Dict].

Therefore, it is not recommended to use this data transformer and send data through columns.
But when specifying columns in the configuration, there will be no problems. Because the order of the columns is determined by the order in the config.

```python
import mlup
from mlup import constants

class Model:
    def predict(self, X):
        return X

columns = [
    {"name": "col1", "type": "int"},
    {"name": "col2", "type": "str"},
    {"name": "col3", "type": "str"},
]
obj1 = [1, 2, 3]
obj2 = [4, 5, 6]
data_without_columns = [obj1, obj2]
data_with_columns = [
    {c["name"]: v for c, v in zip(columns[::-1], obj1[::-1])}, 
    {c["name"]: v for c, v in zip(columns[::-1], obj2[::-1])}
]

# With columns in config
up = mlup.UP(
    ml_model=Model(),
    conf=mlup.Config(
        columns=columns,
        data_transformer_for_predict=constants.ModelDataTransformerType.NUMPY_ARR,
        data_transformer_for_predicted=constants.ModelDataTransformerType.NUMPY_ARR,
    )
)
up.ml.load()
print(up.predict(X=data_with_columns))
# [[1, 2, 3], [4, 5, 6]]
print(up.predict(X=data_without_columns))
# [[1, 2, 3], [4, 5, 6]]

up.conf.columns = None
up.ml.load(force_loading=True)

print(up.predict(X=data_with_columns))
# [[3, 2, 1], [6, 5, 4]]
print(up.predict(X=data_without_columns))
# [[1, 2, 3], [4, 5, 6]]
```

For this data transformer, when creating numpy.array, you can specify dtype. To do this, use the `dtype_for_predict` configuration parameter.
In this parameter, you can specify the dtype name string in numpy. Here are some of them: `float32`, `float64`, `int64`, `int32`, `int16`, `bool_`.

If you do not specify the `dtype_for_predict` parameter in the configuration, then numpy.array determines the types itself.

To reverse the conversion, call `numpy.array(...).tolist()`.

### mlup.ml.data_transformers.tf_tensor_data_transformer.TFTensorDataTransformer

This data transformer transforms data into tensorflow.Tensor, which is used in tensorflow models.
The operating principle and algorithm are similar to `mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer`.
Only the final conversion from `numpy.array` to `tensorflow.convert_to_tensor` is different.

For this data transformer, when creating tensorflow.Tensor, you can specify dtype. To do this, use the `dtype_for_predict` configuration parameter.
In this parameter, you can specify the dtype name string in tensorflow. Here are some of them: `float32`, `float64`, `int64`, `int32`, `int16`, `bool`.

If you do not specify the `dtype_for_predict` parameter in the configuration, then tensorflow.Tensor determines the types itself.

To reverse the conversion, call `tensorflow.Tensor(...).numpy().tolist()`.

### mlup.ml.data_transformers.torch_tensor_data_transformer.TorchTensorDataTransformer

This data transformer transforms the data into torch.Tensor, which is used in torch models.
The operating principle and algorithm are similar to `mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer`.
Only the final conversion from `numpy.array` to `torch.tensor` is different.

For this data transformer, when creating torch.Tensor, you can specify dtype. To do this, use the `dtype_for_predict` configuration parameter.
In this parameter, you can specify the dtype name string in tensorflow. Here are some of them: `float32`, `float64`, `int64`, `int32`, `int16`, `bool`.

If you do not specify the `dtype_for_predict` parameter in the configuration, then torch.Tensor determines the types itself.

To reverse the conversion, call `torch.Tensor(...).tolist()`.

### mlup.ml.data_transformers.src_data_transformer.SrcDataTransformer

This transformer does not do any conversions. There are scenarios when the model accepts python types or JSON valid data types.
Also, the model can return python data or valid JSON data.

For such scenarios this data transformer is used.

## Custom data transformer

If the capabilities of mlup data transformers are not enough for you, you can write your own data transformer.

The data transformer interface is very simple:
```python
# my_module.py
from dataclasses import dataclass
from typing import Any, Optional, List, Union, Dict
from mlup.ml.data_transformers.base import BaseDataTransformer


@dataclass
class MyDataTransformer(BaseDataTransformer):
    dtype_name: Optional[str] = None
    dtype: Optional[Any] = None

    def transform_to_model_format(
        self,
        data: List[Union[Dict[str, Any], List[Any]]],
        columns: Optional[List[Dict[str, str]]] = None,
    ):
        pass

    def transform_to_json_format(self, data: Any):
        pass
```

And specify the path to import your module in `data_transformer_for_predict`: `my_module.MyDataTransformer` or `data_transformer_for_predicted`: `my_module.MyDataTransformer`.

**IMPORTANT: a data transformer written independently must be available for import on the server on which you run the mlup application.**

The easiest way to do this is to create your own python library with your data transformers and other useful classes and install it on your server along with the pymlup library.

