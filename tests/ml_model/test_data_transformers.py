import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlup.constants import ModelDataType
from mlup.ml_model.data_transformers import PandasDataFrameTransformer, NumpyDataFrameTransformer, get_data_transformer_by_type


logger = logging.getLogger('MLupTests')


@pytest.mark.parametrize(
    'data_type, expected_class', [
        (ModelDataType.PANDAS_DF, PandasDataFrameTransformer),
        (ModelDataType.NUMPY_ARR, NumpyDataFrameTransformer),
    ]
)
def test_get_data_transformer_by_type(data_type, expected_class):
    assert get_data_transformer_by_type(data_type) == expected_class


def test_get_data_transformer_by_type_bad_type():
    try:
        get_data_transformer_by_type('not exists model type')
        pytest.fail('Not raised KeyError with not exists type')
    except KeyError:
        pass


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
        df = self.transformer_class.transform_to_model_format(data)

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
        df = self.transformer_class.transform_to_model_format(data, columns=cols)

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
        trans_data = self.transformer_class.transform_to_json_format(df)

        assert trans_data == data


class TestNumpyDataFrameTransformer:
    transformer_class = NumpyDataFrameTransformer

    def test_transform_to_model_format_from_list(self):
        data = [
            [11, 21, 31, 41, 51],
            [12, 22, 32, 42, 52],
            [13, 23, 33, 43, 53],
            [14, 24, 34, 44, 54],
            [15, 25, 35, 45, 55],
        ]
        pred_d = self.transformer_class.transform_to_model_format(data)
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
        pred_d = self.transformer_class.transform_to_model_format(data, columns=cols)
        assert np.array_equal(pred_d, np.array([list(v.values()) for v in data]))

        # Check order by columns
        pred_d = self.transformer_class.transform_to_model_format(data, columns=cols[::-1])
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
        pred_d = self.transformer_class.transform_to_model_format(data)
        assert np.array_equal(pred_d, np.array([list(v.values()) for v in data]))

        # Check order by first item
        reversed_data = data.copy()
        reversed_data[0] = {k: v for k, v in reversed(reversed_data[0].items())}
        pred_d = self.transformer_class.transform_to_model_format(reversed_data)
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

        pred_d = self.transformer_class.transform_to_json_format(arr)
        assert pred_d == data
