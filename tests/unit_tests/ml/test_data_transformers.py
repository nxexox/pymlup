import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlup.constants import ModelDataTransformerType
from mlup.ml.data_transformers.src_data_transformer import SrcDataTransformer
from mlup.utils.interspection import get_class_by_path


logger = logging.getLogger('mlup.test')

try:
    import numpy
    from mlup.ml.data_transformers.numpy_data_transformer import NumpyDataTransformer
except (ImportError, AttributeError) as e:
    logger.info(f'Numpy not installed. Skip tests. {e}')
    numpy, NumpyDataTransformer = None, None

try:
    import pandas
    from mlup.ml.data_transformers.pandas_data_transformer import PandasDataFrameTransformer
except (ImportError, AttributeError) as e:
    logger.info(f'Pandas not installed. Skip tests. {e}')
    pandas, PandasDataFrameTransformer = None, None

try:
    import tensorflow
    from mlup.ml.data_transformers.tf_tensor_data_transformer import TFTensorDataTransformer
    assert_tf_tensors = tensorflow.test.TestCase().assertAllEqual
except (ImportError, AttributeError) as e:
    logger.info(f'tensorflow not installed. Skip tests. {e}')
    tensorflow, TFTensorDataTransformer = None, None

    def assert_tf_tensors(a, b, msg):
        assert False

try:
    import torch
    from mlup.ml.data_transformers.torch_tensor_data_transformer import TorchTensorDataTransformer
    is_equal_torch_tensors = torch.equal
except (ImportError, AttributeError) as e:
    logger.info(f'PyTorch not installed. Skip tests. {e}')
    torch, TorchTensorDataTransformer = None, None

    def is_equal_torch_tensors(a, b):
        return False


@pytest.mark.parametrize(
    'data_type, expected_class', [
        (ModelDataTransformerType.SRC_TYPES, SrcDataTransformer),
        (ModelDataTransformerType.PANDAS_DF, PandasDataFrameTransformer),
        (ModelDataTransformerType.NUMPY_ARR, NumpyDataTransformer),
        (ModelDataTransformerType.TENSORFLOW_TENSOR, TFTensorDataTransformer),
        (ModelDataTransformerType.TORCH_TENSOR, TorchTensorDataTransformer),
    ]
)
def test_get_data_transformer_by_type(data_type, expected_class):
    if expected_class is None:
        return
    assert get_class_by_path(data_type) == expected_class


def test_get_data_transformer_by_type_bad_type():
    try:
        get_class_by_path('not exists transformer type')
        pytest.fail('Not raised KeyError with not exists type')
    except ModuleNotFoundError as e:
        assert str(e) == "No module named 'not exists transformer type'"


def test_get_data_transformer_by_custom_type():
    data = get_class_by_path('mlup.constants.ModelDataTransformerType')
    assert issubclass(data, ModelDataTransformerType)


class TestSrcDataTransformer:
    transformer_class = SrcDataTransformer

    def test_transform_to_model_format_from_list(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        pred_d = self.transformer_class().transform_to_model_format(data)
        assert pred_d == data

    def test_transform_to_model_format_from_dict_with_columns(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'col4': 41, 'col5': 51},
            {'col1': 12, 'col2': 22, 'col3': 32, 'col4': 42, 'col5': 52},
            {'col1': 13, 'col2': 23, 'col3': 33, 'col4': 43, 'col5': 53},
            {'col1': 14, 'col2': 24, 'col3': 34, 'col4': 44, 'col5': 54},
            {'col1': 15, 'col2': 25, 'col3': 35, 'col4': 45, 'col5': 55},
        ]
        cols = [
            {'name': 'col1', 'type': 'int'},
            {'name': 'col2', 'type': 'int'},
            {'name': 'col3', 'type': 'int'},
            {'name': 'col4', 'type': 'int'},
            {'name': 'col5', 'type': 'int'},
        ]
        # Check create
        pred_d = self.transformer_class().transform_to_model_format(data, columns=cols)
        assert pred_d == data

        # Check order by columns
        pred_d = self.transformer_class().transform_to_model_format(data, columns=cols[::-1])
        assert pred_d == data

    def test_transform_to_model_format_from_dict_without_columns(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'col4': 41, 'col5': 51},
            {'col1': 12, 'col2': 22, 'col3': 32, 'col4': 42, 'col5': 52},
            {'col1': 13, 'col2': 23, 'col3': 33, 'col4': 43, 'col5': 53},
            {'col1': 14, 'col2': 24, 'col3': 34, 'col4': 44, 'col5': 54},
            {'col1': 15, 'col2': 25, 'col3': 35, 'col4': 45, 'col5': 55},
        ]
        # Check create
        pred_d = self.transformer_class().transform_to_model_format(data)
        assert pred_d == data

        # Check order by first item
        reversed_data = data.copy()
        reversed_data[0] = {k: v for k, v in reversed(list(reversed_data[0].items()))}
        pred_d = self.transformer_class().transform_to_model_format(reversed_data)
        assert pred_d == reversed_data

    def test_transform_to_json_format(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        pred_d = self.transformer_class().transform_to_json_format(data)
        assert pred_d == data


@pytest.mark.skipif(PandasDataFrameTransformer is None, reason='pandas library not installed.')
class TestPandasDataFrameTransformer:
    transformer_class = PandasDataFrameTransformer

    def test_transform_to_model_format_from_dict(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'colstr': 'colstr', 'colbool': True},
            {'col1': 12, 'col2': 22, 'col3': 32, 'colstr': 'colstr', 'colbool': True},
            {'col1': 13, 'col2': 23, 'col3': 33, 'colstr': 'colstr', 'colbool': False},
            {'col1': 14, 'col2': 24, 'col3': 34, 'colstr': 'colstr', 'colbool': True},
            {'col1': 15, 'col2': 25, 'col3': 35, 'colstr': 'colstr', 'colbool': True},
        ]
        df = self.transformer_class().transform_to_model_format(data)

        assert_frame_equal(df, pd.DataFrame(data=data))
        assert df.columns.tolist() == list(data[0].keys())

    def test_transform_to_model_format_from_list(self):
        cols = [
            {'name': 'col1', 'type': 'int'},
            {'name': 'col2', 'type': 'int'},
            {'name': 'col3', 'type': 'int'},
            {'name': 'colstr', 'type': 'str'},
            {'name': 'colbool', 'type': 'bool'},
        ]
        data = [
            [11, 21, 31, 'colstr', True],
            [12, 22, 32, 'colstr', True],
            [13, 23, 33, 'colstr', False],
            [14, 24, 34, 'colstr', True],
            [15, 25, 35, 'colstr', True],
        ]
        df = self.transformer_class().transform_to_model_format(data, columns=cols)

        data_for_df = [
            {c['name']: v for c, v in zip(cols, d)}
            for d in data
        ]
        assert_frame_equal(df, pd.DataFrame(data=data_for_df))
        assert df.columns.tolist() == [c['name'] for c in cols]

    def test_transform_to_json_format(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'colstr': 'colstr', 'colbool': True},
            {'col1': 12, 'col2': 22, 'col3': 32, 'colstr': 'colstr', 'colbool': True},
            {'col1': 13, 'col2': 23, 'col3': 33, 'colstr': 'colstr', 'colbool': False},
            {'col1': 14, 'col2': 24, 'col3': 34, 'colstr': 'colstr', 'colbool': True},
            {'col1': 15, 'col2': 25, 'col3': 35, 'colstr': 'colstr', 'colbool': True},
        ]
        df = pd.DataFrame(data=data)
        trans_data = self.transformer_class().transform_to_json_format(df)

        assert trans_data == data

    @pytest.mark.parametrize(
        'dtype_name, dtype',
        [
            ('Float32Dtype', pandas.Float32Dtype),
            ('Float64Dtype', pandas.Float64Dtype),
            ('Int8Dtype', pandas.Int8Dtype),
            ('Int16Dtype', pandas.Int16Dtype),
            ('Int32Dtype', pandas.Int32Dtype),
            ('Int64Dtype', pandas.Int64Dtype),
            ('StringDtype', pandas.StringDtype),
            ('BooleanDtype', pandas.BooleanDtype),
        ]
    )
    def test_transform_with_different_dtype(self, dtype_name: str, dtype):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'colbool': False},
            {'col1': 12, 'col2': 22, 'col3': 32, 'colbool': False},
            {'col1': 13, 'col2': 23, 'col3': 33, 'colbool': False},
            {'col1': 14, 'col2': 24, 'col3': 34, 'colbool': True},
            {'col1': 15, 'col2': 25, 'col3': 35, 'colbool': True},
        ]
        if dtype_name == 'BooleanDtype':
            data = [{k: bool(v) for k, v in d.items()} for d in data]
        trans_data = self.transformer_class(dtype_name=dtype_name).transform_to_model_format(data)
        assert_frame_equal(trans_data, pd.DataFrame(data=data, dtype=dtype()))


@pytest.mark.skipif(NumpyDataTransformer is None, reason='numpy library not installed.')
class TestNumpyDataFrameTransformer:
    transformer_class = NumpyDataTransformer

    def test_transform_to_model_format_from_list(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        pred_d = self.transformer_class().transform_to_model_format(data)
        assert np.array_equal(pred_d, np.array(data))

    def test_transform_to_model_format_from_dict_with_columns(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'col4': 41, 'col5': 51},
            {'col1': 12, 'col2': 22, 'col3': 32, 'col4': 42, 'col5': 52},
            {'col1': 13, 'col2': 23, 'col3': 33, 'col4': 43, 'col5': 53},
            {'col1': 14, 'col2': 24, 'col3': 34, 'col4': 44, 'col5': 54},
            {'col1': 15, 'col2': 25, 'col3': 35, 'col4': 45, 'col5': 55},
        ]
        cols = [
            {'name': 'col1', 'type': 'int'},
            {'name': 'col2', 'type': 'int'},
            {'name': 'col3', 'type': 'int'},
            {'name': 'col4', 'type': 'int'},
            {'name': 'col5', 'type': 'int'},
        ]
        # Check create
        pred_d = self.transformer_class().transform_to_model_format(data, columns=cols)
        assert np.array_equal(pred_d, np.array([list(v.values()) for v in data]))

        # Check order by columns
        pred_d = self.transformer_class().transform_to_model_format(data, columns=cols[::-1])
        assert np.array_equal(pred_d, np.array([list(v.values())[::-1] for v in data]))

    def test_transform_to_model_format_from_dict_without_columns(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'col4': 41, 'col5': 51},
            {'col1': 12, 'col2': 22, 'col3': 32, 'col4': 42, 'col5': 52},
            {'col1': 13, 'col2': 23, 'col3': 33, 'col4': 43, 'col5': 53},
            {'col1': 14, 'col2': 24, 'col3': 34, 'col4': 44, 'col5': 54},
            {'col1': 15, 'col2': 25, 'col3': 35, 'col4': 45, 'col5': 55},
        ]
        # Check create
        pred_d = self.transformer_class().transform_to_model_format(data)
        assert np.array_equal(pred_d, np.array([list(v.values()) for v in data]))

        # Check order by first item
        reversed_data = data.copy()
        reversed_data[0] = {k: v for k, v in reversed(list(reversed_data[0].items()))}
        pred_d = self.transformer_class().transform_to_model_format(reversed_data)
        assert np.array_equal(pred_d, np.array([list(v.values())[::-1] for v in data]))

    def test_transform_to_json_format(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        arr = np.array(data)

        pred_d = self.transformer_class().transform_to_json_format(arr)
        assert pred_d == data

    @pytest.mark.parametrize(
        'dtype_name, dtype',
        [
            ('float64', numpy.float64),
            ('float32', numpy.float32),
            ('float16', numpy.float16),
            ('int64', numpy.int64),
            ('int32', numpy.int32),
            ('int16', numpy.int16),
            ('bool_', numpy.bool_),
        ]
    )
    def test_transform_with_different_dtype(self, dtype_name: str, dtype: numpy.generic):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        pred_d = self.transformer_class(dtype_name=dtype_name).transform_to_model_format(data)
        assert np.array_equal(pred_d, np.array(data, dtype=dtype))


@pytest.mark.skipif(TFTensorDataTransformer is None, reason='tensorflow library not installed.')
class TestTensorFlowTensorDataTransformer:
    transformer_class = TFTensorDataTransformer

    def test_transform_to_model_format_from_list(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        pred_d = self.transformer_class().transform_to_model_format(data)
        assert_tf_tensors(pred_d, tensorflow.convert_to_tensor(data))

    def test_transform_to_model_format_from_dict_with_columns(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'col4': 41, 'col5': 51},
            {'col1': 12, 'col2': 22, 'col3': 32, 'col4': 42, 'col5': 52},
            {'col1': 13, 'col2': 23, 'col3': 33, 'col4': 43, 'col5': 53},
            {'col1': 14, 'col2': 24, 'col3': 34, 'col4': 44, 'col5': 54},
            {'col1': 15, 'col2': 25, 'col3': 35, 'col4': 45, 'col5': 55},
        ]
        cols = [
            {'name': 'col1', 'type': 'int'},
            {'name': 'col2', 'type': 'int'},
            {'name': 'col3', 'type': 'int'},
            {'name': 'col4', 'type': 'int'},
            {'name': 'col5', 'type': 'int'},
        ]
        # Check create
        pred_d = self.transformer_class().transform_to_model_format(data, columns=cols)
        assert_tf_tensors(pred_d, tensorflow.convert_to_tensor([list(v.values()) for v in data]))

        # Check order by columns
        pred_d = self.transformer_class().transform_to_model_format(data, columns=cols[::-1])
        assert_tf_tensors(pred_d, tensorflow.convert_to_tensor([list(v.values())[::-1] for v in data]))

    def test_transform_to_model_format_from_dict_without_columns(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'col4': 41, 'col5': 51},
            {'col1': 12, 'col2': 22, 'col3': 32, 'col4': 42, 'col5': 52},
            {'col1': 13, 'col2': 23, 'col3': 33, 'col4': 43, 'col5': 53},
            {'col1': 14, 'col2': 24, 'col3': 34, 'col4': 44, 'col5': 54},
            {'col1': 15, 'col2': 25, 'col3': 35, 'col4': 45, 'col5': 55},
        ]
        # Check create
        pred_d = self.transformer_class().transform_to_model_format(data)
        assert_tf_tensors(pred_d, tensorflow.convert_to_tensor([list(v.values()) for v in data]))

        # Check order by first item
        reversed_data = data.copy()
        reversed_data[0] = {k: v for k, v in reversed(list(reversed_data[0].items()))}
        pred_d = self.transformer_class().transform_to_model_format(reversed_data)
        assert_tf_tensors(pred_d, tensorflow.convert_to_tensor([list(v.values())[::-1] for v in data]))

    def test_transform_to_json_format(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        arr = tensorflow.convert_to_tensor(data)

        pred_d = self.transformer_class().transform_to_json_format(arr)
        assert pred_d == data

    @pytest.mark.parametrize(
        'dtype_name, dtype',
        [
            ('float16', tensorflow.float16),
            ('float32', tensorflow.float32),
            ('float64', tensorflow.float64),
            ('int16', tensorflow.int16),
            ('int32', tensorflow.int32),
            ('int64', tensorflow.int64),
            ('bool', tensorflow.bool),
        ]
    )
    def test_transform_with_different_dtype(self, dtype_name: str, dtype: tensorflow.DType):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        if dtype_name == 'bool':
            data = [[bool(i) for i in r] for r in data]
        pred_d = self.transformer_class(dtype_name=dtype_name).transform_to_model_format(data)
        assert_tf_tensors(pred_d, tensorflow.convert_to_tensor(data, dtype=dtype))


@pytest.mark.skipif(TorchTensorDataTransformer is None, reason='pytorch library not installed.')
class TestTorchTensorDataTransformer:
    transformer_class = TorchTensorDataTransformer

    def test_transform_to_model_format_from_list(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        pred_d = self.transformer_class().transform_to_model_format(data)
        assert is_equal_torch_tensors(pred_d, torch.tensor(data))

    def test_transform_to_model_format_from_dict_with_columns(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'col4': 41, 'col5': 51},
            {'col1': 12, 'col2': 22, 'col3': 32, 'col4': 42, 'col5': 52},
            {'col1': 13, 'col2': 23, 'col3': 33, 'col4': 43, 'col5': 53},
            {'col1': 14, 'col2': 24, 'col3': 34, 'col4': 44, 'col5': 54},
            {'col1': 15, 'col2': 25, 'col3': 35, 'col4': 45, 'col5': 55},
        ]
        cols = [
            {'name': 'col1', 'type': 'int'},
            {'name': 'col2', 'type': 'int'},
            {'name': 'col3', 'type': 'int'},
            {'name': 'col4', 'type': 'int'},
            {'name': 'col5', 'type': 'int'},
        ]
        # Check create
        pred_d = self.transformer_class().transform_to_model_format(data, columns=cols)
        assert is_equal_torch_tensors(pred_d, torch.tensor([list(v.values()) for v in data]))

        # Check order by columns
        pred_d = self.transformer_class().transform_to_model_format(data, columns=cols[::-1])
        assert is_equal_torch_tensors(pred_d, torch.tensor([list(v.values())[::-1] for v in data]))

    def test_transform_to_model_format_from_dict_without_columns(self):
        data = [
            {'col1': 11, 'col2': 21, 'col3': 31, 'col4': 41, 'col5': 51},
            {'col1': 12, 'col2': 22, 'col3': 32, 'col4': 42, 'col5': 52},
            {'col1': 13, 'col2': 23, 'col3': 33, 'col4': 43, 'col5': 53},
            {'col1': 14, 'col2': 24, 'col3': 34, 'col4': 44, 'col5': 54},
            {'col1': 15, 'col2': 25, 'col3': 35, 'col4': 45, 'col5': 55},
        ]
        # Check create
        pred_d = self.transformer_class().transform_to_model_format(data)
        assert is_equal_torch_tensors(pred_d, torch.tensor([list(v.values()) for v in data]))

        # Check order by first item
        reversed_data = data.copy()
        reversed_data[0] = {k: v for k, v in reversed(list(reversed_data[0].items()))}
        pred_d = self.transformer_class().transform_to_model_format(reversed_data)
        assert is_equal_torch_tensors(pred_d, torch.tensor([list(v.values())[::-1] for v in data]))

    def test_transform_to_json_format(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        arr = torch.tensor(data)

        pred_d = self.transformer_class().transform_to_json_format(arr)
        assert pred_d == data

    @pytest.mark.parametrize(
        'dtype_name, dtype',
        [
            ('float16', torch.float16),
            ('float32', torch.float32),
            ('float64', torch.float64),
            ('int16', torch.int16),
            ('int32', torch.int32),
            ('int64', torch.int64),
            ('bool', torch.bool),
        ]
    )
    def test_transform_with_different_dtype(self, dtype_name: str, dtype: torch.dtype):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        if dtype_name == 'bool':
            data = [[bool(i) for i in r] for r in data]
        pred_d = self.transformer_class(dtype_name=dtype_name).transform_to_model_format(data)
        is_equal_torch_tensors(pred_d, torch.tensor(data, dtype=dtype))
