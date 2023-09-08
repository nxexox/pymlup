from typing import List

import pytest

from mlup.constants import IS_X, LoadedFile
from mlup.utils.interspection import analyze_method_params, auto_search_binarization_type


def pred_func_with_X_List(wt, x: List, b: bool = False): pass
def pred_func_with_X_List_of_str(wt, x: List[str], b: bool = False): pass
def pred_func_with_list(wt, x: list, b: bool = False): pass
def pred_func_with_list_of_int(wt, x: list[int], b: bool = False): pass
def pred_func_without_x(wt, y: List, b: bool = False): pass


pred_func_args_without_auto_detect_predict_params = [
    {'name': 'wt', 'required': True},
    {'name': 'x', 'required': True},
    {'name': 'b', 'required': False, 'default': False, 'type': 'bool'},
]
pred_func_args_with_auto_detect_predict_params = [
    {'name': 'wt', 'required': True},
    {'name': 'x', 'required': True, 'type': 'List', IS_X: True},
    {'name': 'b', 'required': False, 'default': False, 'type': 'bool'},
]
pred_func_args_with_auto_detect_predict_params_without_x = [
    {'name': 'wt', 'required': True, 'type': 'List', IS_X: True},
    {'name': 'y', 'required': True},
    {'name': 'b', 'required': False, 'default': False, 'type': 'bool'},
]


class TestAnalyzeMethodParams:

    class ModelClass:
        def pred_func_with_self_and_X_List(self, wt, x: List, b: bool = False): pass
        @classmethod
        def pred_func_with_cls_and_X_List_of_str(cls, wt, x: List[str], b: bool = False): pass
        @staticmethod
        def pred_func_staticmethod_with_list(wt, x: list, b: bool = False): pass
        def pred_func_without_x(self, wt, y: List, b: bool = False): pass

    @pytest.mark.parametrize(
        'pred_func, expected_result',
        [
            pytest.param(
                pred_func_with_X_List, pred_func_args_without_auto_detect_predict_params,
                id='func_with_X_type_List'
            ),
            pytest.param(
                pred_func_with_X_List_of_str, pred_func_args_without_auto_detect_predict_params,
                id='func_with_X_type_List[str]'
            ),
            pytest.param(
                pred_func_with_list, pred_func_args_without_auto_detect_predict_params,
                id='func_with_X_type_list'
            ),
            pytest.param(
                pred_func_with_list_of_int, pred_func_args_without_auto_detect_predict_params,
                id='func_with_X_type_list[int]'
            ),
            pytest.param(
                ModelClass().pred_func_with_self_and_X_List, pred_func_args_without_auto_detect_predict_params,
                id='method_from_obj_with_X_type_List'
            ),
            pytest.param(
                ModelClass.pred_func_with_self_and_X_List, pred_func_args_without_auto_detect_predict_params,
                id='method_fro_cls_with_X_type_List'
            ),
            pytest.param(
                ModelClass.pred_func_with_cls_and_X_List_of_str, pred_func_args_without_auto_detect_predict_params,
                id='classmethod_from_cls_with_cls_and_X_List[str]'
            ),
            pytest.param(
                ModelClass().pred_func_with_cls_and_X_List_of_str, pred_func_args_without_auto_detect_predict_params,
                id='classmethod_from_obj_with_cls_and_X_List[str]'
            ),
            pytest.param(
                ModelClass.pred_func_staticmethod_with_list, pred_func_args_without_auto_detect_predict_params,
                id='staticmethod_from_cls_with_cls_and_X_list'
            ),
            pytest.param(
                ModelClass().pred_func_staticmethod_with_list, pred_func_args_without_auto_detect_predict_params,
                id='staticmethod_from_obj_with_cls_and_X_list'
            ),
        ],
    )
    def test_without_auto_detect_predict_params(self, pred_func, expected_result):
        inspection_params = analyze_method_params(pred_func, auto_detect_predict_params=False)
        assert inspection_params == expected_result

    @pytest.mark.parametrize(
        'pred_func, expected_result',
        [
            pytest.param(
                pred_func_with_X_List, pred_func_args_with_auto_detect_predict_params,
                id='func_with_X_type_List'
            ),
            pytest.param(
                pred_func_with_X_List_of_str, pred_func_args_with_auto_detect_predict_params,
                id='func_with_X_type_List[str]'
            ),
            pytest.param(
                pred_func_with_list, pred_func_args_with_auto_detect_predict_params,
                id='func_with_X_type_list'
            ),
            pytest.param(
                pred_func_with_list_of_int, pred_func_args_with_auto_detect_predict_params,
                id='func_with_X_type_list[int]'
            ),
            pytest.param(
                ModelClass().pred_func_with_self_and_X_List, pred_func_args_with_auto_detect_predict_params,
                id='method_from_obj_with_X_type_List'
            ),
            pytest.param(
                ModelClass.pred_func_with_self_and_X_List, pred_func_args_with_auto_detect_predict_params,
                id='method_from_cls_with_X_type_List'
            ),
            pytest.param(
                ModelClass.pred_func_with_cls_and_X_List_of_str, pred_func_args_with_auto_detect_predict_params,
                id='classmethod_from_cls_with_X_type_List[str]'
            ),
            pytest.param(
                ModelClass().pred_func_with_cls_and_X_List_of_str, pred_func_args_with_auto_detect_predict_params,
                id='classmethod_from_obj_with_X_type_List[str]'
            ),
            pytest.param(
                ModelClass.pred_func_staticmethod_with_list, pred_func_args_with_auto_detect_predict_params,
                id='staticmethod_from_cls_with_X_type_list'
            ),
            pytest.param(
                ModelClass().pred_func_staticmethod_with_list, pred_func_args_with_auto_detect_predict_params,
                id='staticmethod_from_obj_with_X_type_list'
            ),
            pytest.param(
                pred_func_without_x, pred_func_args_with_auto_detect_predict_params_without_x,
                id='func_without_X'
            ),
            pytest.param(
                ModelClass.pred_func_without_x, pred_func_args_with_auto_detect_predict_params_without_x,
                id='method_from_cls_without_X'
            ),
            pytest.param(
                ModelClass().pred_func_without_x, pred_func_args_with_auto_detect_predict_params_without_x,
                id='method_from_obj_without_X'
            ),
        ],
    )
    def test_with_auto_detect_predict_params(self, pred_func, expected_result):
        inspection_params = analyze_method_params(pred_func, auto_detect_predict_params=True)
        assert inspection_params == expected_result

    @pytest.mark.parametrize(
        'pred_func, expected_result',
        [
            pytest.param(
                ModelClass.pred_func_with_self_and_X_List,
                [{'name': 'self', 'required': True}] + pred_func_args_without_auto_detect_predict_params,
                id='method_with_X_List'
            ),
            pytest.param(
                ModelClass.pred_func_with_cls_and_X_List_of_str, pred_func_args_without_auto_detect_predict_params,
                id='classmethod_from_cls_with_X_List[str]'
            ),
            pytest.param(
                ModelClass.pred_func_staticmethod_with_list, pred_func_args_without_auto_detect_predict_params,
                id='staticmethod_from_cls_with_X_list'
            ),
            pytest.param(
                ModelClass.pred_func_without_x,
                [
                    {'name': 'self', 'required': True},
                    {'name': 'wt', 'required': True},
                    {'name': 'y', 'required': True},
                    {'name': 'b', 'required': False, 'default': False, 'type': 'bool'},
                ],
                id='method_from_cls_without_X'
            ),
        ],
    )
    def test_without_ignore_self(self, pred_func, expected_result):
        inspection_params = analyze_method_params(pred_func, auto_detect_predict_params=False, ignore_self=False)
        assert inspection_params == expected_result


class TestAutoSearchBinarizationType:
    @pytest.mark.parametrize(
        'loaded_file',
        [
            LoadedFile(),
            LoadedFile(''),
            LoadedFile(path=''),
            LoadedFile('not valid bytes', 'not valid path.test'),
            LoadedFile(path='joblib.joblib')
        ],
        ids=['empty', 'empty_raw', 'empty_path', 'not_valid_data', 'joblib']
    )
    def test_not_found_binarizer(self, loaded_file):
        assert auto_search_binarization_type(loaded_file) is None

    @pytest.mark.parametrize(
        'loaded_file',
        [
            LoadedFile(path='pickle.pckl'),
            LoadedFile(path='model.onnx'),
            LoadedFile(b'PK\x03', path='model.h5'),
            LoadedFile(b'PK\x03', path='model.pth'),
            LoadedFile('\n\n\n\nversion=\n', path='.txt')
        ],
        ids=['pickle', 'onnx', 'tensorflow', 'torch', 'lightgbm'],
    )
    def test_found_binarizer(self, loaded_file):
        assert auto_search_binarization_type(loaded_file) is not None
