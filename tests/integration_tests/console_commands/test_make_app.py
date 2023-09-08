import logging
import os
import pickle
import signal
import subprocess
import time

import pytest
import requests

import mlup
from mlup.console_scripts.make_app import make_app
from mlup.constants import ModelDataTransformerType


logger = logging.getLogger('mlup.test')


@pytest.mark.parametrize(
    'custom_params',
    [{}, {'port': 8010}],
    ids=['without_custom_params', 'with_custom_params']
)
def test_run_maked_app_from_conf(tmp_path_factory, pickle_print_model_config_yaml, custom_params):
    path_to_app = str(tmp_path_factory.getbasetemp() / 'test_run_maked_app_from_conf' /
                      'test_run_maked_app_from_conf.py')
    os.makedirs(os.path.dirname(path_to_app), exist_ok=True)
    make_app(
        path_to_file=path_to_app,
        path_to_src=pickle_print_model_config_yaml,
        use_config=True,
        config_fields=custom_params,
        force=True,
    )
    proc = subprocess.Popen(
        'python ' + path_to_app,
        shell=True,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    port = custom_params.get('port', mlup.Config.port)
    i = 0
    response, error = None, None
    while i <= 10 and response is None:
        try:
            response = requests.post(f'http://0.0.0.0:{port}/predict', json={'X': [[1, 2, 3]]})
        except requests.ConnectionError:
            time.sleep(i)
        except Exception as e:
            error = e
        i += 1

    # SIGTERM
    os.kill(proc.pid, signal.SIGINT)
    if error:
        raise error

    output = proc.stderr.read()
    logger.info(output)
    assert response.status_code == 200
    assert response.json() == {'predict_result': [[1, 2, 3]]}


@pytest.mark.parametrize(
    'custom_params',
    [{}, {'port': 8011}],
    ids=['without_custom_params', 'with_custom_params']
)
def test_run_maked_app_from_up_bin(tmp_path_factory, print_model, custom_params):
    path_to_app = str(tmp_path_factory.getbasetemp() / 'test_run_maked_app_from_up_bin' /
                      'test_run_maked_app_from_up_bin.py')
    path_to_pickle = os.path.join(os.path.dirname(path_to_app), 'test_run_maked_app_from_up_bin.pckl')
    os.makedirs(os.path.dirname(path_to_app), exist_ok=True)

    up = mlup.UP(
        ml_model=print_model,
        conf=mlup.Config(port=8010, data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    up.ml.load()
    with open(path_to_pickle, 'wb') as f:
        pickle.dump(up, f)

    make_app(
        path_to_file=path_to_app,
        path_to_src=path_to_pickle,
        use_binary=True,
        config_fields=custom_params,
        force=True,
    )
    proc = subprocess.Popen(
        'python ' + path_to_app,
        shell=True,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    port = custom_params.get('port', up.conf.port)
    i = 0
    response, error = None, None
    while i <= 10 and response is None:
        try:
            response = requests.post(f'http://0.0.0.0:{port}/predict', json={'X': [[1, 2, 3]]})
        except requests.ConnectionError:
            time.sleep(i)
        except Exception as e:
            error = e
        i += 1

    # SIGTERM
    os.kill(proc.pid, signal.SIGINT)
    if error:
        raise error

    output = proc.stderr.read()
    logger.info(output)
    assert response.status_code == 200
    assert response.json() == {'predict_result': [[1, 2, 3]]}


@pytest.mark.parametrize(
    'custom_params',
    [
        {},
        {
            'port': 8012,
            'data_transformer_for_predict': f"'{ModelDataTransformerType.NUMPY_ARR.value}'"
        }
    ],
    ids=[
        'without_custom_params',
        'with_custom_params'
    ]
)
def test_run_maked_app_from_model_bin(tmp_path_factory, scikit_learn_binary_cls_model, custom_params):
    path_to_app = str(tmp_path_factory.getbasetemp() / 'test_run_maked_app_from_model_bin' /
                      'test_run_maked_app_from_model_bin.py')
    os.makedirs(os.path.dirname(path_to_app), exist_ok=True)

    make_app(
        path_to_file=path_to_app,
        path_to_src=scikit_learn_binary_cls_model.path,
        use_model=True,
        config_fields=custom_params,
        force=True,
    )
    proc = subprocess.Popen(
        'python ' + path_to_app,
        shell=True,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    port = custom_params.get('port', mlup.Config.port)
    test_data_raw = scikit_learn_binary_cls_model.test_data_raw
    if custom_params:
        test_data_raw = [1, 2, 3, 4, 5, 6, 7, 8]

    i = 0
    response, error = None, None
    while i <= 10 and response is None:
        try:
            response = requests.post(
                f'http://0.0.0.0:{port}/predict',
                json={scikit_learn_binary_cls_model.x_arg_name: [test_data_raw]}
            )
        except requests.ConnectionError:
            time.sleep(i)
        except Exception as e:
            error = e
        i += 1

    # SIGTERM
    os.kill(proc.pid, signal.SIGINT)
    if error:
        raise error

    output = proc.stderr.read()
    logger.info(output)
    assert response.status_code == 200
    assert response.json() == {"predict_result": [scikit_learn_binary_cls_model.test_model_response_raw]}
