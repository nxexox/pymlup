import abc
import logging
from typing import List, Dict, Any, Union, Type, Optional

import numpy
from numpy import ndarray
from pandas import DataFrame

from mlup.constants import ModelDataType


logger = logging.getLogger('MLup')


class BaseDataTransformer(metaclass=abc.ABCMeta):
    """Base class for transformers data for predict to model format and back."""
    DATA_TYPE: ModelDataType

    @classmethod
    @abc.abstractmethod
    def transform_to_model_format(
        cls,
        data: List[Union[Dict[str, Any], List[Any]]],
        columns: Optional[List[Dict[str, str]]] = None,
    ):
        pass

    @classmethod
    @abc.abstractmethod
    def transform_to_json_format(cls, data: Union[ndarray, DataFrame, Any]):
        pass


class PandasDataFrameTransformer(BaseDataTransformer):
    """Class for transformers data for predict to DataFrame format and back."""
    DATA_TYPE: ModelDataType = ModelDataType.PANDAS_DF

    @classmethod
    def transform_to_model_format(
        cls,
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

        df = DataFrame(data=_data_dict)
        logger.debug(f'Create pandas.DataFrame {df.shape} success.')
        return df

    @classmethod
    def transform_to_json_format(cls, data: DataFrame):
        return data.to_dict('records')


class NumpyDataFrameTransformer(BaseDataTransformer):
    """Class for transformers data for predict to numpy.ndarray format and back."""
    DATA_TYPE: ModelDataType = ModelDataType.NUMPY_ARR

    @classmethod
    def transform_to_model_format(
        cls,
        data: List[Union[Dict[str, Any], List[Any]]],
        columns: Optional[List[Dict[str, str]]] = None,
    ) -> numpy.ndarray:
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
        return numpy.array(result)

    @classmethod
    def transform_to_json_format(cls, data: ndarray):
        return data.tolist()


data_transformers: Dict[ModelDataType, Type[BaseDataTransformer]] = {
    ModelDataType.PANDAS_DF: PandasDataFrameTransformer,
    ModelDataType.NUMPY_ARR: NumpyDataFrameTransformer,
}


def get_data_transformer_by_type(
    data_type: Union[str, ModelDataType],
) -> Union[
    Type[PandasDataFrameTransformer],
    Type[NumpyDataFrameTransformer],
]:
    if isinstance(data_type, str):
        try:
            data_type = ModelDataType(data_type)
        except ValueError as e:
            raise KeyError(e)

    try:
        return data_transformers[data_type]
    except KeyError:
        logger.error(f'DataTransformer type {data_type} not supported.')
        raise
