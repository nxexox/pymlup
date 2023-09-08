import logging
from typing import Any, List, Dict, Union, Optional

import pandas
from pandas import DataFrame
from pandas._typing import Dtype

from mlup.constants import ModelDataTransformerType
from mlup.ml.data_transformers.base import BaseDataTransformer


logger = logging.getLogger('mlup')


class PandasDataFrameTransformer(BaseDataTransformer):
    """Class for transform data for predict to DataFrame format and back."""
    dtype: Optional[Dtype] = None
    DATA_TYPE: ModelDataTransformerType = ModelDataTransformerType.PANDAS_DF

    def get_dtype(self) -> Dtype:
        return getattr(pandas, self.dtype_name)()

    def transform_to_model_format(
        self,
        data: List[Union[Dict[str, Any], List[Any]]],
        columns: Optional[List[Dict[str, str]]] = None,
    ) -> DataFrame:
        _data_dict: Dict[str, List] = dict()
        for obj in data:
            if isinstance(obj, list):
                if not columns:
                    raise ValueError('If input data have type list, than need set columns.')
                for column_conf, value in zip(columns, obj):
                    column = column_conf["name"]
                    _data_dict.setdefault(column, [])
                    _data_dict[column].append(value)
            else:
                for column, value in obj.items():
                    _data_dict.setdefault(column, [])
                    _data_dict[column].append(value)

        df = DataFrame(data=_data_dict, dtype=self.dtype)
        logger.debug(f'Create pandas.DataFrame {df.shape} success.')
        return df

    def transform_to_json_format(self, data: DataFrame):
        return data.to_dict('records')
