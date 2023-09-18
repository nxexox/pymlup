import json
import os
import pickle
import time
from multiprocessing import Process

import pytest
import requests

from mlup.constants import BinarizationType
from mlup.errors import ModelLoadError, ModelBinarizationError

try:
    import joblib
except ImportError:
    joblib = None

import mlup
from mlup.console_scripts.run import run, run_from_up_bin, run_from_config, run_from_model, InputArgsError


@pytest.mark.parametrize(
    'config_path, config_type',
    [('not_exists_path.json', 'json'), ('not_exists_path.yaml', 'yaml')],
    ids=['json', 'yaml']
)
def test_run_from_config_not_exists_conf(config_path, config_type):
    try:
        run_from_config(config_path, config_type)
        pytest.fail('Not raised error')
    except FileNotFoundError as e:
        assert str(e) == f"[Errno 2] No such file or directory: '{config_path}'"


def test_run_from_config_not_exists_conf_type(tmp_path_factory):
    file_path = tmp_path_factory.getbasetemp() / 'test_run_from_config_not_exists_conf_type' / \
                'test_run_from_config_not_exists_conf_type.json'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('Not valid config\n\nBut is valid multirows string.')

    try:
        run_from_config(str(file_path), 'not_exists_type')
        pytest.fail('Not raised error')
    except ValueError as e:
        assert str(e) == 'Config type not_exists_type not supported.'


@pytest.mark.parametrize('config_type', ['json', 'yaml'])
def test_run_from_config_not_valid_config(tmp_path_factory, config_type):
    file_path = tmp_path_factory.getbasetemp() / 'test_run_from_config_not_valid_config' / \
                f'test_run_from_config_not_valid_config.{config_type}'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('Not valid config\n\nBut is valid multirows string.')

    try:
        run_from_config(str(file_path), config_type)
        pytest.fail('Not raised error')
    except json.JSONDecodeError as e:
        assert str(e) == 'Expecting value: line 1 column 1 (char 0)'
    except AttributeError as e:
        assert str(e) == "'str' object has no attribute 'get'"


@pytest.mark.parametrize(
    'config_fixture_name, conf_type',
    [('test_json_config', 'json'), ('test_yaml_config', 'yaml')],
    ids=['json', 'yaml']
)
@pytest.mark.asyncio
async def test_run_from_config_valid_config(request, pickle_print_model, config_fixture_name, conf_type):
    config_path = request.getfixturevalue(config_fixture_name)
    if conf_type == 'json':
        _up = mlup.UP.load_from_json(config_path, load_model=False)
        _up.conf.uvicorn_kwargs['loop'] = 'none'
        _up.conf.storage_kwargs['path_to_files'] = str(pickle_print_model)
        _up.to_json(config_path)
    elif conf_type == 'yaml':
        _up = mlup.UP.load_from_yaml(config_path, load_model=False)
        _up.conf.uvicorn_kwargs['loop'] = 'none'
        _up.conf.storage_kwargs['path_to_files'] = str(pickle_print_model)
        _up.to_yaml(config_path)
    else:
        pytest.fail(f'Not supported config type {conf_type}')

    proc = Process(
        target=run_from_config,
        args=(config_path, conf_type),
        daemon=False
    )
    try:
        proc.start()
        time.sleep(10)

        resp = requests.get('http://0.0.0.0:8009/health')
        assert resp.status_code == 200
        assert resp.json() == {'status': 200}
    finally:
        proc.terminate()
        time.sleep(3)


@pytest.mark.parametrize(
    'binary_path, binary_type',
    [('not_exists_path.pckl', 'pickle'), ('not_exists_path.joblib', 'joblib')],
    ids=['pickle', 'joblib']
)
def test_run_from_up_bin_not_exists_bin(binary_path, binary_type):
    try:
        run_from_up_bin(binary_path, binary_type)
        pytest.fail('Not raised error')
    except FileNotFoundError as e:
        assert str(e) == f"[Errno 2] No such file or directory: '{binary_path}'"


def test_run_from_up_bin_not_exists_binary_type(tmp_path_factory):
    file_path = tmp_path_factory.getbasetemp() / 'test_run_from_up_bin_not_exists_binary_type' / \
                'test_run_from_up_bin_not_exists_binary_type.pckl'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('Not valid binary\n\nBut is valid multirows string.')

    try:
        run_from_up_bin(str(file_path), 'not_exists_type')
        pytest.fail('Not raised error')
    except ValueError as e:
        assert str(e) == 'Binary type not_exists_type not supported.'


@pytest.mark.parametrize('binary_type', ['pickle', 'joblib'])
def test_run_from_up_bin_not_valid_binary(tmp_path_factory, binary_type):
    file_path = tmp_path_factory.getbasetemp() / 'test_run_from_up_bin_not_valid_binary' / \
                f'test_run_from_up_bin_not_valid_binary.{binary_type}'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('Not valid binary data\n\nBut is valid multirows string.')

    try:
        run_from_up_bin(str(file_path), binary_type)
        pytest.fail('Not raised error')
    except pickle.UnpicklingError as e:
        assert str(e) == 'could not find MARK'
    except IndexError as e:
        assert str(e) == 'pop from empty list'


@pytest.mark.asyncio
async def test_run_from_up_bin_valid_binary_pickle(tmp_path_factory, print_model):
    full_path = str(tmp_path_factory.getbasetemp() / 'test_run_from_up_bin_valid_binary_pickle' /
                    'test_run_from_up_bin_valid_binary_pickle.pickle')
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    up = mlup.UP(ml_model=print_model)
    up.ml.load()
    up.conf.uvicorn_kwargs['loop'] = 'none'
    with open(full_path, 'wb') as f:
        pickle.dump(up, f)

    proc = Process(
        target=run_from_up_bin,
        args=(full_path, 'pickle'),
        daemon=False
    )
    try:
        proc.start()
        time.sleep(10)

        resp = requests.get('http://0.0.0.0:8009/health')
        assert resp.status_code == 200
        assert resp.json() == {'status': 200}
    finally:
        proc.terminate()
        time.sleep(3)


@pytest.mark.skipif(joblib is None, reason='joblib library not installed.')
@pytest.mark.asyncio
async def test_run_from_up_bin_valid_binary_joblib(tmp_path_factory, print_model):
    full_path = str(tmp_path_factory.getbasetemp() / 'test_run_from_up_bin_valid_binary_joblib' /
                    'test_run_from_bin_valid_binary_joblib.joblib')
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    up = mlup.UP(ml_model=print_model)
    up.ml.load()
    up.conf.uvicorn_kwargs['loop'] = 'none'
    joblib.dump(up, full_path)

    proc = Process(
        target=run_from_up_bin,
        args=(full_path, 'joblib'),
        daemon=False
    )
    try:
        proc.start()
        time.sleep(10)

        resp = requests.get('http://0.0.0.0:8009/health')
        assert resp.status_code == 200
        assert resp.json() == {'status': 200}
    finally:
        proc.terminate()
        time.sleep(3)


def test_run_from_model_not_exists_model():
    try:
        run_from_model('not_exists_model.pckl')
        pytest.fail('Not raised error')
    except ModelLoadError as e:
        assert str(e) == "[Errno 2] No such file or directory: 'not_exists_model.pckl'"


def test_run_from_model_not_valid_model(tmp_path_factory):
    file_path = tmp_path_factory.getbasetemp() / 'test_run_from_model_not_valid_model' / \
                'test_run_from_model_not_valid_model.pckl'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write('Not valid model\n\nBut is valid multirows string.')

    try:
        run_from_model(str(file_path), binarization_type=BinarizationType.PICKLE)
        pytest.fail('Not raised error')
    except ModelBinarizationError as e:
        assert str(e) == 'Error with deserialize model: could not find MARK'


@pytest.mark.asyncio
async def test_run_from_model_valid_binary_pickle(pickle_print_model):
    proc = Process(
        target=run_from_model,
        kwargs={'model_path': pickle_print_model, 'port': 8011},
        daemon=False
    )
    try:
        proc.start()
        time.sleep(20)

        resp = requests.get('http://0.0.0.0:8011/health')
        assert resp.status_code == 200
        assert resp.json() == {'status': 200}
    finally:
        proc.terminate()
        time.sleep(3)


def test_run_with_conf_and_bin():
    try:
        run(
            path='',
            use_conf=True,
            use_bin=True
        )
        pytest.fail('Not raised error')
    except InputArgsError as e:
        assert str(e) == 'You can specify only one of the arguments: use_conf [-c], use_bin [-b], use_model [-m].'


def test_run_without_conf_and_bin():
    try:
        run(
            path='',
            use_conf=False,
            use_bin=False,
        )
    except InputArgsError as e:
        assert str(e) == 'You must specify one of the arguments: use_conf [-c], use_bin [-b], use_model [-m].'


def test_run_with_not_exists_path():
    try:
        run(
            path='nox_exists_path',
            use_conf=True,
        )
    except FileNotFoundError as e:
        assert str(e) == 'Path nox_exists_path not exists.'
