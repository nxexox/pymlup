import asyncio
import logging
import pickle
import threading
import time
from typing import List, Dict

import numpy as np
import pytest

from mlup.constants import ModelDataTransformerType, DEFAULT_X_ARG_NAME, StorageType, BinarizationType
from mlup.errors import ModelLoadError, PredictTransformDataError
from mlup.ml.model import MLupModel, ModelConfig
from mlup.utils.loop import create_async_task

logger = logging.getLogger('mlup.test')


class ModelWithX:
    def predict(self, X: List):
        return X


@pytest.fixture(scope="session")
def pickle_with_x_model(tmp_path_factory):
    model = ModelWithX()
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'pickle_easy_model_for_test_model.pckl'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        pickle.dump(model, f)
    return f_path / model_name


class TestMLupModelPublicMethods:
    @pytest.mark.asyncio
    async def test_no_load_model(self):
        mlup_model = MLupModel(conf=ModelConfig(storage_type=StorageType.disk))
        try:
            await mlup_model.predict(X=[[1, 2, 3]])
            pytest.fail('Not raised ModelLoadError in not loaded model.')
        except ModelLoadError as e:
            assert str(e) == 'Model object not loaded. Please call load().'

    def test_not_exists_predict_method(self, pickle_with_x_model):
        ml_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_with_x_model},
                predict_method_name='not_exists_method'
            )
        )
        try:
            ml_model.load()
            pytest.fail('Not raised ModelLoadError, if not exists predict method.')
        except ModelLoadError as e:
            assert str(e) == "'ModelWithX' object has no attribute 'not_exists_method'"

    @pytest.mark.asyncio
    async def test_load_model_from_pickle_file(self, pickle_with_x_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_with_x_model},
            )
        )
        mlup_model.load()
        await mlup_model.predict(X=[[1, 2, 3]])

    @pytest.mark.asyncio
    async def test_load_model_from_obj(self):
        ml_model = ModelWithX()
        # With default stprage_type=memory
        mlup_model = MLupModel(
            ml_model=ml_model,
            conf=ModelConfig(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        mlup_model.load()
        await mlup_model.predict(X=[[1, 2, 3]])

        # With storage_type=memory
        mlup_model = MLupModel(
            ml_model=ml_model,
            conf=ModelConfig(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
                storage_type=StorageType.memory,
            )
        )
        mlup_model.load()
        await mlup_model.predict(X=[[1, 2, 3]])

    @pytest.mark.asyncio
    async def test_load_model_from_obj_ignore_binarizer(self):
        model = ModelWithX()
        mlup_model = MLupModel(
            ml_model=model,
            conf=ModelConfig(
                storage_type=StorageType.memory,
                binarization_type=BinarizationType.JOBLIB,
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        mlup_model.load()
        assert model == mlup_model.model_obj
        pred_res = await mlup_model.predict(X=[[1, 2, 3]])
        assert pred_res == [[1, 2, 3]]

    @pytest.mark.parametrize(
        'force_loading, expected, default',
        [
            pytest.param(False, 'test', None, id='force_loading=False'),
            pytest.param(True, 'attr not found', 'attr not found', id='force_loading=True')
        ]
    )
    def test_loading_binary_model_from_pickle_with_force_loading(
        self, pickle_print_model, pickle_print_sleep_model, force_loading: bool, expected: str, default
    ):
        mlup_model = MLupModel(
            conf=ModelConfig(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model}
            )
        )
        mlup_model.load()

        # Set attribute
        mlup_model.model_obj.TestAttribute = 'test'
        assert getattr(mlup_model.model_obj, 'TestAttribute') == 'test'

        # Reloading model
        mlup_model.conf.storage_kwargs['path_to_files'] = pickle_print_sleep_model
        try:
            mlup_model.load(force_loading=force_loading)
        except ModelLoadError as e:
            assert str(e) == 'Model is already loaded. For reload use force_reloading=True.'
            assert force_loading is False
        assert getattr(mlup_model.model_obj, 'TestAttribute', default) == expected

    @pytest.mark.parametrize(
        'force_loading, expected, default',
        [
            pytest.param(False, 'test', None, id='force_loading=False'),
            pytest.param(True, 'attr not found', 'attr not found', id='force_loading=True')
        ]
    )
    def test_loading_binary_model_from_obj_with_force_loading(
        self, force_loading: bool, expected: str, default
    ):
        ml_model = ModelWithX()
        mlup_model = MLupModel(
            ml_model=ml_model,
            conf=ModelConfig(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        mlup_model.load()

        # Set attribute
        mlup_model.model_obj.TestAttribute = 'test'
        assert getattr(mlup_model.model_obj, 'TestAttribute') == 'test'

        # Reloading my model
        ml_model = ModelWithX()
        mlup_model.conf.storage_kwargs['model'] = ml_model
        try:
            mlup_model.load(force_loading=force_loading)
        except ModelLoadError as e:
            assert str(e) == 'Model is already loaded. For reload use force_reloading=True.'
            assert force_loading is False
        assert getattr(mlup_model.model_obj, 'TestAttribute', default) == expected

    def test_load_model_error(self):
        try:
            MLupModel()
            pytest.fail('Not raised ModelLoadError, if not set path to model.')
        except ModelLoadError as e:
            assert str(e) == f'If you use {StorageType.memory}, need set "ml" argument.'

    def test_load_model_from_pickle_not_exist_file(self, pickle_not_exists_file):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_not_exists_file},
            )
        )
        try:
            mlup_model.load()
            pytest.fail('Not raised ModelLoadError, if not exists file.')
        except ModelLoadError as e:
            assert str(e).startswith('[Errno 2] No such file or directory: ')

    def test_load_model_from_pickle_bad_reload(self, pickle_print_model, pickle_print_sleep_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
            )
        )
        mlup_model.load()
        mlup_model.conf.storage_kwargs['path_to_files'] = pickle_print_sleep_model
        try:
            mlup_model.load(force_loading=False)
            pytest.fail('Not raise ModelLoadError, because not use force_loading=True')
        except ModelLoadError as e:
            assert str(e) == 'Model is already loaded. For reload use force_reloading=True.'

    def test_reload_binary_model_from_pickle_change_path(self, pickle_print_model, pickle_print_sleep_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model}
            )
        )
        mlup_model.load()

        # Check not reload
        setattr(mlup_model.model_obj, 'test', 'test')
        mlup_model.load(force_loading=True)
        assert hasattr(mlup_model.model_obj, 'test') is False
        src_model = mlup_model.model_obj

        # Check reload model
        mlup_model.conf.storage_kwargs['path_to_files'] = pickle_print_sleep_model
        mlup_model.load(force_loading=True)
        assert hasattr(mlup_model.model_obj, 'test') is False
        assert id(src_model) != id(mlup_model.model_obj)

    def test_load_model_from_pickle_change_predict_method(self, pickle_print_model, pickle_print_sleep_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model}
            )
        )
        mlup_model.load()
        predict_method = mlup_model.model_obj.predict
        assert mlup_model._predict_method == predict_method

        # Check not reload
        mlup_model.conf.storage_kwargs['path_to_files'] = pickle_print_sleep_model
        try:
            mlup_model.load()
            pytest.fail('Not raise ModelLoadError, because not use force_loading=True')
        except ModelLoadError:
            assert mlup_model._predict_method == predict_method

        # Check reload model
        mlup_model.load(force_loading=True)
        assert mlup_model._predict_method != predict_method
        assert mlup_model._predict_method == mlup_model.model_obj.predict

        # Check change predict_method_name
        predict_method = mlup_model._predict_method
        mlup_model.conf.predict_method_name = 'second_predict'
        mlup_model.load(force_loading=True)
        assert mlup_model._predict_method != predict_method

    def test_load_model_from_pickle_change_x_column_name(self, pickle_print_model, pickle_print_sleep_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
            )
        )
        mlup_model.load()
        x_column_name = mlup_model.x_column_name
        assert mlup_model.x_column_name == x_column_name

        # Check not reload
        try:
            mlup_model.load()
            pytest.fail('Not raise ModelLoadError, because not use force_loading=True')
        except ModelLoadError:
            assert mlup_model.x_column_name == x_column_name

        # Check reload model
        mlup_model.conf.storage_kwargs['path_to_files'] = pickle_print_sleep_model
        mlup_model.load(force_loading=True)
        assert mlup_model.x_column_name == x_column_name

        # Check change predict_method_name
        x_column_name = mlup_model.x_column_name
        mlup_model.conf.predict_method_name = 'predict_with_x_name_y'
        mlup_model.load(force_loading=True)
        assert mlup_model.x_column_name != x_column_name
        assert mlup_model.x_column_name == 'Y'

    def test_load_model_from_pickle_change_predict_arguments(
        self, pickle_with_x_model, pickle_print_sleep_model
    ):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_with_x_model},
            )
        )
        mlup_model.load()
        predict_arguments = mlup_model._predict_arguments
        assert mlup_model._predict_arguments == predict_arguments

        # Check not reload
        try:
            mlup_model.load()
            pytest.fail('Not raise ModelLoadError, because not use force_loading=True')
        except ModelLoadError:
            assert mlup_model._predict_arguments == predict_arguments

        # Check reload model
        mlup_model.conf.storage_kwargs['path_to_files'] = pickle_print_sleep_model
        mlup_model.load(force_loading=True)
        assert mlup_model._predict_arguments != predict_arguments

        # Check change predict_method_name
        predict_arguments = mlup_model._predict_arguments
        mlup_model.conf.predict_method_name = 'predict_with_x_name_y'
        mlup_model.load(force_loading=True)
        assert mlup_model._predict_arguments != predict_arguments
        assert mlup_model._predict_arguments == [
            {'name': 'Y', 'required': True, 'collection_type': 'List', 'is_X': True},
            {'name': 'test_param', 'required': False, 'type': 'bool', 'default': False},
        ]

    def test_load_model_from_pickle_change_use_thread_loop(self, pickle_print_model, pickle_print_sleep_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
            )
        )
        mlup_model.load()
        assert mlup_model.conf.use_thread_loop is True
        assert mlup_model._pool_workers is not None

        # Check not reload
        try:
            mlup_model.load()
            pytest.fail('Not raise ModelLoadError, because not use force_loading=True')
        except ModelLoadError:
            assert mlup_model.conf.use_thread_loop is True
            assert mlup_model._pool_workers is not None

        # Check reload model
        mlup_model.conf.storage_kwargs['path_to_files'] = pickle_print_sleep_model
        mlup_model.load(force_loading=True)
        assert mlup_model.conf.use_thread_loop is True
        assert mlup_model._pool_workers is not None

        # Check change use_thread_loop
        mlup_model.conf.use_thread_loop = False
        mlup_model.load(force_loading=True)
        assert mlup_model.conf.use_thread_loop is False
        assert mlup_model._pool_workers is None

    def test_load_model_settings_without_model(self):
        mlup_model = MLupModel(conf=ModelConfig(storage_type=StorageType.disk))
        try:
            mlup_model.load_model_settings()
            pytest.fail('Not raised ModelLoadError in not loaded model.')
        except ModelLoadError as e:
            assert str(e) == 'Model not loaded to memory. Analyze impossible. Please call ml.load().'

    @pytest.mark.parametrize(
        'kwargs',
        [
            pytest.param({}, id='default_values'),
            pytest.param(dict(predict_method_name='predict'), id='not_default_values')
        ]
    )
    def test_load_model_settings_change_predict_method_name(self, pickle_print_model, kwargs: Dict):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                **kwargs,
            )
        )
        mlup_model.load()
        src_predict_method = mlup_model._predict_method

        mlup_model.conf.predict_method_name = 'second_predict'
        mlup_model.load_model_settings()
        assert src_predict_method != mlup_model._predict_method

    @pytest.mark.parametrize(
        'kwargs',
        [
            pytest.param({}, id='default_values'),
            pytest.param(dict(predict_method_name='predict'), id='not_default_values')
        ]
    )
    def test_load_model_settings_change_predict_arguments(self, pickle_print_model, kwargs: Dict):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                **kwargs,
            )
        )
        mlup_model.load()
        src_predict_arguments = mlup_model._predict_arguments

        mlup_model.conf.predict_method_name = 'predict_with_x_name_y'
        mlup_model.load_model_settings()
        assert src_predict_arguments != mlup_model._predict_arguments

    @pytest.mark.parametrize(
        'kwargs',
        [
            pytest.param({}, id='default_values'),
            pytest.param(
                dict(predict_method_name='predict', auto_detect_predict_params=True),
                id='not_default_values'
            )
        ]
    )
    def test_load_model_settings_change_x_column_name(self, pickle_print_model, kwargs: Dict):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                **kwargs,
            )
        )
        mlup_model.load()
        src_x_column_name = mlup_model.x_column_name

        # Check if change method, to method with other signature
        mlup_model.conf.predict_method_name = 'predict_with_x_name_y'
        mlup_model.load_model_settings()
        assert src_x_column_name != mlup_model.x_column_name

        new_x_column_name = mlup_model.x_column_name
        # Check if set auto_detect_predict_params=False
        mlup_model.conf.auto_detect_predict_params = False
        mlup_model.load_model_settings()
        assert mlup_model.x_column_name != src_x_column_name
        assert mlup_model.x_column_name != new_x_column_name
        assert mlup_model.x_column_name == DEFAULT_X_ARG_NAME

    @pytest.mark.parametrize(
        'kwargs',
        [
            pytest.param({}, id='default_values'),
            pytest.param(dict(use_thread_loop=True), id='not_default_values')
        ]
    )
    def test_load_model_settings_change_use_thread_loop(self, pickle_print_model, kwargs: Dict):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                **kwargs,
            )
        )
        mlup_model.load()
        src_pool_workers = mlup_model._pool_workers

        mlup_model.conf.use_thread_loop = False
        mlup_model.load_model_settings()
        assert mlup_model._pool_workers is None
        assert src_pool_workers != mlup_model._pool_workers

    def test_get_X_from_predict_data_not_loaded_model(self, pickle_print_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model}
            )
        )
        try:
            mlup_model.get_X_from_predict_data({})
            pytest.fail('Not raised ModelLoadError in not loaded model.')
        except ModelLoadError as e:
            assert str(e) == 'Model object not loaded. Please call load().'

    @pytest.mark.parametrize(
        'auto_detect_predict_params, data, expected',
        [
            pytest.param(False, {'X': [1, 2, 3]}, None, id='auto_detect_predict_params=False,key=X'),
            pytest.param(
                False, {'test': [1, 2, 3]}, None,
                id='auto_detect_predict_params=False,key=test'
            ),
            pytest.param(
                False, {DEFAULT_X_ARG_NAME: [1, 2, 3]}, [1, 2, 3],
                id='auto_detect_predict_params=False,key=DEFAULT_X_ARG_NAME'
            ),
            pytest.param(True, {'X': [1, 2, 3]}, [1, 2, 3], id='auto_detect_predict_params=True,key=X'),
            pytest.param(True, {'test': [1, 2, 3]}, None, id='auto_detect_predict_params=True,key=test'),
            pytest.param(
                True, {DEFAULT_X_ARG_NAME: [1, 2, 3]}, None,
                id='auto_detect_predict_params=True,key=DEFAULT_X_ARG_NAME'
            ),
        ]
    )
    @pytest.mark.parametrize('remove', [False, True], ids=lambda x: f'remove={x}')
    def test_get_X_from_predict_data_default_column_name(
        self, pickle_print_model, auto_detect_predict_params: bool, data: Dict, expected, remove: bool
    ):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                auto_detect_predict_params=auto_detect_predict_params,
            )
        )
        mlup_model.load()
        pred_col = mlup_model.get_X_from_predict_data({'test': [1, 2, 3]})
        assert pred_col is None

        pred_col = mlup_model.get_X_from_predict_data(data, remove=remove)
        assert pred_col == expected
        if remove:
            assert mlup_model.x_column_name not in data

    @pytest.mark.asyncio
    async def test_predict_not_loaded_model(self, pickle_print_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
            )
        )
        try:
            await mlup_model.predict(X=[[1, 2, 3]])
        except ModelLoadError as e:
            assert str(e) == 'Model object not loaded. Please call load().'

    @pytest.mark.asyncio
    async def test_easy_predict(self, pickle_print_model):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        mlup_model.load()
        pred_result = await mlup_model.predict(X=[[1, 2, 3]])
        assert pred_result == [[1, 2, 3]]

    @pytest.mark.parametrize(
        'kwargs, data, error_msg',
        [
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.PANDAS_DF,
                 'data_transformer_for_predicted': ModelDataTransformerType.PANDAS_DF},
                [[1, 2, 3]],
                'If input data have type list, than need set columns.',
                id='LIST-TO-DF-FROM-DF'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.PANDAS_DF,
                 'data_transformer_for_predicted': ModelDataTransformerType.PANDAS_DF},
                [1, 2, 3],
                "'int' object has no attribute 'items'",
                id='PRIMITIVE-TO-DF-FROM-DF'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.NUMPY_ARR,
                 'data_transformer_for_predicted': ModelDataTransformerType.PANDAS_DF},
                [1, 2, 3],
                "'int' object has no attribute 'items'",
                id='PRIMITIVE-TO-NP-FROM-DF'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.NUMPY_ARR,
                 'data_transformer_for_predicted': ModelDataTransformerType.PANDAS_DF},
                [[1, 2, 3]],
                "'numpy.ndarray' object has no attribute 'to_dict'",
                id='PRIMITIVE-TO-NP-FROM-DF'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.PANDAS_DF,
                 'data_transformer_for_predicted': ModelDataTransformerType.NUMPY_ARR},
                [{'col1': 1, 'col2': 2, 'col3': 3}],
                "'DataFrame' object has no attribute 'tolist'",
                id='DICTS-TO-DF-FROM-NP'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.PANDAS_DF,
                 'data_transformer_for_predicted': ModelDataTransformerType.NUMPY_ARR,
                 'columns': [{'name': 'col1'}, {'name': 'col2'}, {'name': 'col3'}]},
                [[1, 2, 3]],
                "'DataFrame' object has no attribute 'tolist'",
                id='DICTS-TO-DF-FROM-NP-with_columns'
            ),
        ]
    )
    @pytest.mark.asyncio
    async def test_predict_with_not_correct_data_transformers(
        self, pickle_print_model, kwargs: Dict, data, error_msg
    ):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                **kwargs
            )
        )
        mlup_model.load()
        try:
            await mlup_model.predict(X=data)
            pytest.fail('Not raised TransformDataError')
        except PredictTransformDataError as e:
            assert str(e) == error_msg

    @pytest.mark.parametrize(
        'kwargs, data, expected_data',
        [
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.PANDAS_DF,
                 'data_transformer_for_predicted': ModelDataTransformerType.PANDAS_DF},
                [{}],
                [],
                id='EMPTY-DICT-TO-DF-FROM-DF'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.PANDAS_DF,
                 'data_transformer_for_predicted': ModelDataTransformerType.PANDAS_DF},
                [],
                [],
                id='EMPTY-TO-DF-FROM-DF'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.PANDAS_DF,
                 'data_transformer_for_predicted': ModelDataTransformerType.PANDAS_DF,
                 'columns': [{'name': 'col1'}, {'name': 'col2'}, {'name': 'col3'}]},
                [[1, 2, 3]],
                [{'col1': 1, 'col2': 2, 'col3': 3}],
                id='LIST-TO-DF-FROM-DF-with_columns_in_construct'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.PANDAS_DF,
                 'data_transformer_for_predicted': ModelDataTransformerType.PANDAS_DF},
                [{'col1': 1, 'col2': 2, 'col3': 3}],
                [{'col1': 1, 'col2': 2, 'col3': 3}],
                id='LIST-TO-DF-FROM-DF-with_columns_in_request'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.NUMPY_ARR,
                 'data_transformer_for_predicted': ModelDataTransformerType.NUMPY_ARR},
                [[1, 2, 3]],
                [[1, 2, 3]],
                id='LIST-TO-NP-FROM-NP'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.NUMPY_ARR,
                 'data_transformer_for_predicted': ModelDataTransformerType.NUMPY_ARR},
                [],
                [],
                id='EMPTY-TO-NP-FROM-NP'
            ),
            pytest.param(
                {'data_transformer_for_predict': ModelDataTransformerType.NUMPY_ARR,
                 'data_transformer_for_predicted': ModelDataTransformerType.NUMPY_ARR},
                [[]],
                [[]],
                id='EMPTY-LIST-TO-NP-FROM-NP'
            ),
        ]
    )
    @pytest.mark.asyncio
    async def test_predict_with_correct_data_transformers(
        self, pickle_print_model, kwargs: Dict, data, expected_data
    ):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                **kwargs
            )
        )
        mlup_model.load()
        pred_data = await mlup_model.predict(X=data)
        assert pred_data == expected_data

    @pytest.mark.asyncio
    async def test_predict_in_thread_pool_executor(self):
        class ThreadIdModel:
            def predict(self, X: List):
                return np.array([threading.current_thread().ident])
        mlup_model = MLupModel(
            ml_model=ThreadIdModel(),
            conf=ModelConfig(
                use_thread_loop=True,
                max_thread_loop_workers=5,
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        mlup_model.load()

        # Check, that code was launched in ThreadPoolExecutor workers.
        thread_id = await mlup_model.predict(X=[[1, 2, 3]])
        thread_pool_ids = {th.ident for th in mlup_model._pool_workers._threads}
        assert thread_id[0] in thread_pool_ids

        # Check, other thread_id, if set use_thread=False
        mlup_model.conf.use_thread_loop = False
        thread_id = await mlup_model.predict(X=[[1, 2, 3]])
        assert thread_id[0] not in thread_pool_ids

    @pytest.mark.asyncio
    async def test_predict_without_thread_pool_executor(self):
        class ThreadIdModel:
            def predict(self, X: List):
                return np.array([threading.current_thread().ident])
        mlup_model = MLupModel(
            ml_model=ThreadIdModel(),
            conf=ModelConfig(
                use_thread_loop=False,
                max_thread_loop_workers=1,
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        mlup_model.load()

        # Check, many predicted in single thread.
        thread_ids = {
            (await mlup_model.predict(X=[[1, 2, 3]]))[0],
            (await mlup_model.predict(X=[[1, 2, 3]]))[0],
            (await mlup_model.predict(X=[[1, 2, 3]]))[0],
            (await mlup_model.predict(X=[[1, 2, 3]]))[0],
            (await mlup_model.predict(X=[[1, 2, 3]]))[0],
            (await mlup_model.predict(X=[[1, 2, 3]]))[0],
            (await mlup_model.predict(X=[[1, 2, 3]]))[0],
        }
        assert mlup_model._pool_workers is None
        assert len(thread_ids) == 1

    @pytest.mark.parametrize(
        'use_thread_loop, max_thread_loop_workers',
        [
            pytest.param(False, 1, id='multiprocessing.Lock'),
            pytest.param(True, 1, id='asyncio.locks.Lock'),
        ]
    )
    @pytest.mark.asyncio
    async def test_predict_lock(
        self, pickle_print_model, pickle_print_sleep_model, use_thread_loop: bool, max_thread_loop_workers: int
    ):
        mlup_model = MLupModel(
            conf=ModelConfig(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': pickle_print_model},
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
                use_thread_loop=use_thread_loop,
                max_thread_loop_workers=max_thread_loop_workers,
            )
        )
        mlup_model.load()

        async def call_predict(mlup_model, data):
            return await mlup_model.predict(**data)

        # Check Lock not sleep model. Fast work
        tasks = []
        start = time.monotonic()
        for i in range(3):
            tasks.append(
                create_async_task(
                    call_predict(mlup_model, {'X': [[1, 2, 3]]}),
                    name=f'test_predict_lock-not_sleep_{i}',
                )
            )
        await asyncio.gather(*tasks)
        assert time.monotonic() - start < 1

        # Check Lock with sleep model. Slowly work
        mlup_model.conf.storage_kwargs['path_to_files'] = pickle_print_sleep_model
        mlup_model.load(force_loading=True)
        mlup_model.model_obj.sleep = 0.35
        tasks = []
        start = time.monotonic()
        for i in range(3):
            tasks.append(
                create_async_task(
                    call_predict(mlup_model, {'X': [[1, 2, 3]]}),
                    name=f'test_predict_lock-with_sleep_{i}'
                )
            )
        await asyncio.gather(*tasks)
        # 1 second per 1 call predict
        assert time.monotonic() - start > 1

    @pytest.mark.asyncio
    async def test_predict_from_numpy(self):
        mlup_model = MLupModel(
            ml_model=ModelWithX(),
            conf=ModelConfig(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
                data_transformer_for_predicted=ModelDataTransformerType.SRC_TYPES,
            )
        )
        mlup_model.load()

        predicted_data = await mlup_model.predict_from(X=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        np.array_equal(predicted_data, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    @pytest.mark.asyncio
    async def test_predict_with_list_columns(self):
        mlup_model = MLupModel(
            ml_model=ModelWithX(),
            conf=ModelConfig(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
                data_transformer_for_predicted=ModelDataTransformerType.SRC_TYPES,
                columns=[
                    {"name": "col1", "type": "int", "collection_type": "List"},
                    {"name": "col2", "type": "int", "collection_type": "List"},
                ],
            )
        )
        mlup_model.load()

        predicted_data = await mlup_model.predict_from(X=[[1, 2], [3, 4], [5, 6]])
        np.array_equal(predicted_data, np.array([[1, 2], [3, 4], [5, 6]]))
