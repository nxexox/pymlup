import logging
from typing import Any, List, Dict, Union, Optional

import tensorflow as tf

from mlup.constants import ModelDataTransformerType
from mlup.ml.data_transformers.base import BaseDataTransformer


logger = logging.getLogger('mlup')


class TFTensorDataTransformer(BaseDataTransformer):
    """Class for transformers data for predict to tensorflow.Tensor format and back."""
    dtype: Optional[tf.DType] = None
    DATA_TYPE: ModelDataTransformerType = ModelDataTransformerType.TENSORFLOW_TENSOR

    def get_dtype(self) -> tf.DType:
        return getattr(tf, self.dtype_name)

    def transform_to_model_format(
        self,
        data: List[Union[Dict[str, Any], List[Any]]],
        columns: Optional[List[Dict[str, str]]] = None,
    ) -> tf.Tensor:
        result = []
        data_type = None
        columns = columns or []
        for obj in data:
            if not data_type:
                if isinstance(obj, list):
                    data_type = 'list'
                else:
                    data_type = 'dict'

            if data_type == 'list':
                result.append(obj)
            else:
                # Order regularization by columns or columns in first object
                _r = []
                if columns:
                    for c_conf in columns:
                        _r.append(obj[c_conf["name"]])
                else:
                    for _k, _v in obj.items():
                        columns.append({"name": _k})
                        _r.append(_v)
                result.append(_r)

        logger.debug(f'Create tensorflow.Tensor {len(result)} success.')
        return tf.convert_to_tensor(result, dtype=self.dtype)

    def transform_to_json_format(cls, data: tf.Tensor):
        return data.numpy().tolist()
