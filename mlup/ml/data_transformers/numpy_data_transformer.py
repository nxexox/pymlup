import logging
from typing import Any, List, Dict, Union, Optional

import numpy as np

from mlup.constants import ModelDataTransformerType
from mlup.ml.data_transformers.base import BaseDataTransformer


logger = logging.getLogger('mlup')


class NumpyDataTransformer(BaseDataTransformer):
    """Class for transformers data for predict to numpy.ndarray format and back."""
    dtype: Optional[np.generic] = None
    DATA_TYPE: ModelDataTransformerType = ModelDataTransformerType.NUMPY_ARR

    def get_dtype(self) -> np.generic:
        return getattr(np, self.dtype_name)

    def transform_to_model_format(
        self,
        data: List[Union[Dict[str, Any], List[Any]]],
        columns: Optional[List[Dict[str, str]]] = None,
    ) -> np.ndarray:
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

        logger.debug(f'Create numpy.ndarray {len(result)} success.')
        return np.array(result, dtype=self.dtype)

    def transform_to_json_format(self, data: np.ndarray):
        return data.tolist()
