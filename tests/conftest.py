import json
import logging
import os
import shutil
import socket
import sys
from dataclasses import dataclass, field
import pickle
import time
from itertools import cycle
from pathlib import Path
from typing import List, Type, Any, Dict

import httpx
import joblib
import numpy as np
import pytest
import yaml

import mlup
from mlup.constants import DEFAULT_X_ARG_NAME, StorageType, ModelDataTransformerType
from mlup.web.app import MLupWebApp


logger = logging.getLogger('mlup.test')


_test_json_config = {
    "ml": {
        "binarization_type": "mlup.ml.binarization.pickle.PickleBinarizer",
        "data_transformer_for_predicted": "mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer",
        "data_transformer_for_predict": "mlup.ml.data_transformers.pandas_data_transformer.PandasDataFrameTransformer",
        "use_thread_loop": True,
        "columns": [
            {"name": "MinTemp", "type": "float", "required": False, "default": 1.4},
            {"name": "MaxTemp", "type": "float", "required": False},
            {"name": "Humidity9am", "type": "float", "required": False},
            {"name": "Humidity3pm", "type": "float", "required": False},
            {"name": "Pressure9am", "type": "float", "required": False},
            {"name": "Pressure3pm", "type": "float", "required": False},
            {"name": "Temp9am", "type": "float", "required": False},
            {"name": "Temp3pm", "type": "float", "required": False}
        ],
        "storage_type": "mlup.ml.storage.local_disk.DiskStorage",
        "storage_kwargs": {
            "path_to_files": "test/path/to/file.pckl",
        },
        "auto_detect_predict_params": True,
        "max_thread_loop_workers": None,
        "type": "sklearn",
        "version": "1.2.3.4",
        "predict_method_name": "predict",
        "name": "Test"
    },
    "web": {
        "host": "0.0.0.0",
        "column_validation": False,
        "mode": "mlup.web.architecture.directly_to_predict.DirectlyToPredictArchitecture",
        "web_app_version": "1.0.0.0",
        "min_batch_len": 10,
        "port": 8009,
        "ttl_predicted_data": 60,
        "throttling_max_requests": None,
        "item_id_col_name": "mlup_item_id",
        "is_long_predict": False,
        "batch_worker_timeout": 1,
        "max_queue_size": 100,
        "throttling_max_request_len": None,
        "timeout_for_shutdown_daemon": 3,
        "debug": False,
        "show_docs": True,
        "ttl_client_wait": 30.0,
        "uvicorn_kwargs": {},
    }
}


@pytest.fixture(scope="session")
def test_dict_config():
    return _test_json_config


@pytest.fixture(scope="session")
def test_yaml_config(tmp_path_factory):
    f_path = tmp_path_factory.getbasetemp()
    config_name = 'yaml_test_config.yaml'
    logger.info(f'Create {f_path}/{config_name} fixture.')
    with open(f_path / config_name, 'w') as f:
        yaml.dump(_test_json_config, f)
    return f_path / config_name


@pytest.fixture(scope="session")
def test_json_config(tmp_path_factory):
    f_path = tmp_path_factory.getbasetemp()
    config_name = 'yaml_test_config.json'
    logger.info(f'Create {f_path}/{config_name} fixture.')
    with open(f_path / config_name, 'w') as f:
        json.dump(_test_json_config, f)
    return f_path / config_name


@dataclass
class TestMlupWebClient:
    """
    This client maked because FastApi client fastapi.testclient.TestClient
    сrashed when used multiple times with multiple applications.

    This class starts web with run(daemon=True).

    """
    app: MLupWebApp
    wait_running: float = 1
    wait_after_shutdown: float = 3

    def __enter__(self):
        self.app.conf.host = 'localhost'
        self.app.conf.port = self.get_free_port()
        self.app.run(daemon=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.app.stop(shutdown_timeout=self.app.conf.timeout_for_shutdown_daemon)
        except Exception as e:
            logger.error(f'It is OK. Shutdown web error, before unicorn.shutdown is stuck: {e}')

    @classmethod
    def get_free_port(cls) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            for port in cycle(range(8009, 8015)):
                r = s.connect_ex(('localhost', port))
                if r != 0:
                    return port

    async def get(self, url, params=None) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            return await client.get(f'http://localhost:{self.app.conf.port}{url}', params=params)

    async def post(self, url, params=None, json=None) -> httpx.Response:
        async with httpx.AsyncClient() as client:
            return await client.post(f'http://localhost:{self.app.conf.port}{url}', params=params, json=json)


@pytest.fixture
def web_app_test_client():
    return TestMlupWebClient


class TestModel:
    def run_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        pass

    def predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        return self.run_predict(X, test_param, *custom_args, **custom_kwargs)

    def second_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        return self.run_predict(X, test_param, *custom_args, **custom_kwargs)

    def predict_with_x_name_y(self, Y: List, test_param: bool = False, *custom_args, **custom_kwargs):
        return self.run_predict(Y, test_param, *custom_args, **custom_kwargs)


class PrintModel(TestModel):
    def run_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        print(
            f'Call PrintModel.run_predict('
            f'X={X}, test_param={test_param}, *custom_args={custom_args}, custom_kwargs={custom_kwargs}'
            f')'
        )
        return X


@dataclass
class PrintSleepModel(TestModel):
    sleep: float = 1

    def run_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        time.sleep(self.sleep)
        print(
            f'Call PrintSleepModel.run_predict('
            f'X={X}, test_param={test_param}, *custom_args={custom_args}, custom_kwargs={custom_kwargs}'
            f')'
        )
        return X


class ListToNumpyArrayModel(TestModel):
    def run_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        print(
            f'Call ListToNumpyArrayModel.run_predict('
            f'X={X}, test_param={test_param}, *custom_args={custom_args}, custom_kwargs={custom_kwargs}'
            f')'
        )
        return np.array(X)


class RaiseExceptionModel(TestModel):
    exc = ValueError('Test message')

    def run_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        raise self.exc


class ModelWithX:
    def predict(self, X: List):
        return X


@pytest.fixture(scope="function")
def print_model() -> PrintModel:
    return PrintModel()


@pytest.fixture(scope="function")
def print_sleep_model() -> PrintSleepModel:
    return PrintSleepModel()


@pytest.fixture(scope="function")
def model_with_x() -> ModelWithX:
    return ModelWithX()


@pytest.fixture(scope="session")
def model_with_x_class() -> Type[ModelWithX]:
    return ModelWithX


@pytest.fixture(scope="function")
def raise_exception_model() -> RaiseExceptionModel:
    return RaiseExceptionModel()


@pytest.fixture(scope="function")
def list_to_numpy_array_model() -> ListToNumpyArrayModel:
    return ListToNumpyArrayModel()


@pytest.fixture(scope="session")
def pickle_print_model(tmp_path_factory):
    model = PrintModel()
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'pickle_print_model.pckl'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        pickle.dump(model, f)
    return f_path / model_name


@pytest.fixture(scope="session")
def pickle_print_sleep_model(tmp_path_factory):
    model = PrintSleepModel(sleep=1)
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'pickle_print_sleep_model.pckl'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        pickle.dump(model, f)
    return f_path / model_name


@pytest.fixture(scope="session")
def joblib_print_model(tmp_path_factory):
    model = PrintModel()
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'joblib_print_model.joblib'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        joblib.dump(model, f)
    return f_path / model_name


@pytest.fixture(scope="session")
def joblib_print_sleep_model(tmp_path_factory):
    model = PrintSleepModel(sleep=1)
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'joblib_print_sleep_model.joblib'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        joblib.dump(model, f)
    return f_path / model_name


@pytest.fixture(scope="session")
def pickle_not_exists_file(tmp_path_factory):
    return tmp_path_factory.getbasetemp() / 'not_exists_file.pckl'


@pytest.fixture(scope="session")
def pickle_not_exists_folder(tmp_path_factory):
    return tmp_path_factory.getbasetemp() / 'not_exists_folder'


@pytest.fixture(scope="session")
def root_dir(request):
    return request.config.rootdir


@pytest.fixture(scope="session")
def models_datadir(root_dir):
    return root_dir / 'mldata' / 'models'


@pytest.fixture(scope="session")
def tests_jupyter_notebooks_datadir():
    return 'tests/integration_tests/notebooks'


@dataclass
class ModelAndPath:
    path: str
    model: Any
    test_data_raw: Dict = field(default_factory=lambda: {
        'MinTemp': 1.0,
        'MaxTemp': 2.0,
        'Humidity9am': 3.0,
        'Humidity3pm': 4.0,
        'Pressure9am': 5.0,
        'Pressure3pm': 6.0,
        'Temp9am': 7.0,
        'Temp3pm': 8.0,
    })
    test_model_response_raw: Any = 'No'
    test_model_response_round: int = 4
    predict_method_name: str = '__call__'
    file_mask: str = r'(\w.-_)*.pckl'
    is_many_files: bool = False
    x_arg_name: str = DEFAULT_X_ARG_NAME


@pytest.fixture(scope="session")
def scikit_learn_binary_cls_model(models_datadir):
    try:
        import sklearn
        with open(models_datadir / 'scikit-learn-binary_cls_model.pckl', 'rb') as f:
            model = pickle.load(f)
        return ModelAndPath(
            models_datadir / 'scikit-learn-binary_cls_model.pckl',
            model,
            test_model_response_raw='No',
            predict_method_name='predict',
            x_arg_name='X',
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def pickle_scikit_learn_model_config_yaml(tmp_path_factory, scikit_learn_binary_cls_model):
    f_path = tmp_path_factory.getbasetemp()
    config_name = 'pickle_scikit_learn_model_conf.yaml'
    logger.info(f'Create {f_path}/{config_name} fixture.')
    up = mlup.UP(
        conf=mlup.Config(
            storage_type=StorageType.disk,
            storage_kwargs={'path_to_files': str(scikit_learn_binary_cls_model.path)},
            data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
        )
    )
    up.to_yaml(f_path / config_name)
    return f_path / config_name


@pytest.fixture(scope="session")
def scikit_learn_binary_cls_model_onnx(models_datadir):
    try:
        import onnxruntime as onnxrt
        from mlup.ml.binarization.onnx import _InferenceSessionWithPredict
        model = _InferenceSessionWithPredict(str(models_datadir / 'scikit-learn-binary_cls_model.onnx'))
        return ModelAndPath(
            models_datadir / 'scikit-learn-binary_cls_model.onnx',
            model,
            test_model_response_raw='No',
            predict_method_name='predict',
            x_arg_name='input_data',
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def lightgbm_binary_cls_model(models_datadir):
    try:
        with open(models_datadir / 'lightgbm-binary_cls_model.pckl', 'rb') as f:
            return ModelAndPath(
                models_datadir / 'lightgbm-binary_cls_model.pckl',
                pickle.load(f),
                test_model_response_raw=0.1978,
                x_arg_name='data',
            )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def lightgbm_binary_cls_model_txt(models_datadir):
    try:
        import lightgbm as lgb
        return ModelAndPath(
            models_datadir / 'lightgbm-binary_cls_model.txt',
            lgb.Booster(model_file=models_datadir / 'lightgbm-binary_cls_model.txt'),
            test_model_response_raw=0.1978,
            x_arg_name='data',
            file_mask=r'(\w.-_)*.txt',
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def tensorflow_binary_cls_model(models_datadir):
    try:
        import tensorflow
        model_name = 'tensorflow-binary_cls_model.pckl'
        model_result = 0.5144
        # For python 3.7
        if sys.version_info.minor == 7:
            model_name = 'tensorflow-binary_cls_model37.pckl'
            model_result = 0.6405

        with open(models_datadir / model_name, 'rb') as f:
            return ModelAndPath(
                models_datadir / model_name,
                pickle.load(f),
                test_model_response_raw=model_result,
            )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def tensorflow_binary_cls_model_keras(models_datadir, tmp_path_factory):
    try:
        import tensorflow
        model_name = 'tensorflow-binary_cls_model.keras'
        model_result = 0.5144
        # For python 3.7
        if sys.version_info.minor == 7:
            model_name = 'tensorflow-binary_cls_model37.keras'
            model_result = 0.6405

        path_to_model = models_datadir / model_name
        model = tensorflow.keras.models.load_model(str(path_to_model), compile=False)
        model.compile()

        return ModelAndPath(
            path_to_model,
            model=model,
            test_model_response_raw=model_result,
            file_mask=Path(path_to_model).name,
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def tensorflow_binary_cls_model_h5(models_datadir):
    try:
        import tensorflow
        model_name = 'tensorflow-binary_cls_model.h5'
        model_result = 0.5144
        # For python 3.7
        if sys.version_info.minor == 7:
            model_name = 'tensorflow-binary_cls_model37.h5'
            model_result = 0.6405

        path_to_model = models_datadir / model_name
        model = tensorflow.keras.models.load_model(path_to_model, compile=False)
        model.compile()

        return ModelAndPath(
            path_to_model,
            model=model,
            test_model_response_raw=model_result,
            file_mask=r'(\w.-_)*.h5',
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def tensorflow_binary_cls_model_zip(models_datadir):
    try:
        import tensorflow
        # Only python3.8+
        return ModelAndPath(
            models_datadir,
            model=tensorflow.saved_model.load(models_datadir / 'tensorflow-binary_cls_model.savedmodel'),
            test_model_response_raw=0.5144,
            file_mask=r'tensorflow-binary_cls_model.savedmodel',
            is_many_files=True,
            predict_method_name='serve',
            x_arg_name='args_0'
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def pytorch_binary_cls_model(models_datadir):
    try:
        import torch
        with open(models_datadir / 'pytorch-binary_cls_model.pckl', 'rb') as f:
            return ModelAndPath(
                models_datadir / 'pytorch-binary_cls_model.pckl',
                pickle.load(f),
                test_model_response_raw=0.4163,
            )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def pytorch_binary_cls_model_onnx(models_datadir):
    try:
        import onnxruntime as onnxrt
        from mlup.ml.binarization.onnx import _InferenceSessionWithPredict
        model = _InferenceSessionWithPredict(str(models_datadir / 'pytorch-binary_cls_model.onnx'))
        return ModelAndPath(
            models_datadir / 'pytorch-binary_cls_model.onnx',
            model,
            test_model_response_raw=0.4163,
            predict_method_name='predict',
            x_arg_name='input',
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def pytorch_binary_cls_model_pth(models_datadir):
    try:
        import torch
        with open(models_datadir / 'pytorch-binary_cls_model.pth', 'rb') as f:
            model = torch.load(f)
            model.eval()
        return ModelAndPath(
            models_datadir / 'pytorch-binary_cls_model.pth',
            model,
            file_mask=r'(\w.-_)*.pth',
            test_model_response_raw=0.4163,
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def pytorch_binary_cls_model_jit(models_datadir):
    try:
        import torch
        import io
        with open(models_datadir / 'pytorch-binary_cls_model-jit.pth', 'rb') as f:
            model = torch.jit.load(io.BytesIO(f.read()))
        model.eval()
        return ModelAndPath(
            models_datadir / 'pytorch-binary_cls_model-jit.pth',
            model,
            file_mask=r'(\w.-_)*.pth',
            test_model_response_raw=0.4163,
        )
    except ImportError:
        return None
