import logging
import os
from pathlib import Path

import pytest

try:
    import joblib
except ImportError:
    joblib = None

from mlup.console_scripts.make_app import (
    make_app, save_app,
    get_app_code_from_up_bin, get_app_code_from_conf, get_app_code_without_preload,
    get_app_code_from_model_bin,
    GenerateCodeError, SaveAppError, InputArgsError,
)


logger = logging.getLogger('mlup.test')


APP_CODE_WITHOUT_PRELOAD_TEMPLATE = """import mlup


# You can load the model yourself and pass it to the "ml_model" argument.
# up = mlup.UP(ml_model=my_model, conf=mlup.Config())
up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        # You can use storage_type and storage_kwargs for auto_load model from storage.{custom_conf}
    )
)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn {path}:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
"""


GET_APP_CODE_FROM_CONF_TEMPLATE = """import mlup


up = mlup.UP.load_from_{config_type}('{config_path}', load_model=False){custom_conf}
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn {path}:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
"""


GET_APP_CODE_FROM_PICKLE_TEMPLATE = """import pickle


with open('{binary_path}', 'rb') as f:
    up = pickle.load(f)
{custom_conf}
if not up.ml.loaded:
    up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn {path}:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
"""


GET_APP_CODE_FROM_JOBLIB_TEMPLATE = """import joblib


up = joblib.load('{binary_path}')
{custom_conf}
if not up.ml.loaded:
    up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn {path}:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
"""


GET_APP_CODE_FROM_MODEL_BIN_TEMPLATE = """import mlup
from mlup import constants


up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        storage_type=constants.StorageType.disk,
        storage_kwargs={{
            'path_to_files': '{model_path}',
            'files_mask': '{model_name}',
        }},{custom_conf}
    )
)
up.ml.load()
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn {path}:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
"""


PATCHES_TO_FILE = [
    'app.py',
    'folder/pup.py',
    'folder/folders/folderssasdasd/test.py/op.py',
    'app',
    'folder/pup',
    'folder/folders/folderssasdasd/test.py/op',
    'app.exe',
    'folder/pup.exe',
    'folder/folders/folderssasdasd/test.py/op.exe',
]
CONF_KWARGS_DICT = {
    'port': 8011,
    'batch_worker_timeout': 10.0,
    'predict_method_name': "'__call__'",
    'use_thread_loop': False,
    'columns': [{"name": "col", "type": "list"}],
    'uvicorn_kwargs': {'workers': 4, 'timeout_graceful_shutdown': 10},
}


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
@pytest.mark.parametrize('with_custom_conf', [False, True])
def test_get_app_code_without_preload(path, with_custom_conf):
    if with_custom_conf:
        kwargs_str = ''
        for k, v in CONF_KWARGS_DICT.items():
            kwargs_str += f"\n        {k}={v},"

        generated_code = get_app_code_without_preload(path, **CONF_KWARGS_DICT)
        expected_code = APP_CODE_WITHOUT_PRELOAD_TEMPLATE.format(
            path=Path(path).stem, custom_conf=kwargs_str
        )
    else:
        generated_code = get_app_code_without_preload(path)
        expected_code = APP_CODE_WITHOUT_PRELOAD_TEMPLATE.format(path=Path(path).stem, custom_conf='')
    assert generated_code == expected_code


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
@pytest.mark.parametrize(
    'config_path, config_type',
    [('not_exists_path.json', 'json'), ('not_exists_path.yaml', 'yaml')],
    ids=['json', 'yaml']
)
def test_get_app_code_from_conf_not_exists_conf(path, config_path, config_type):
    assert get_app_code_from_conf(path, config_path, config_type) == GET_APP_CODE_FROM_CONF_TEMPLATE.format(
        path=Path(path).stem,
        config_path=config_path,
        config_type=config_type,
        custom_conf='',
    )


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
def test_get_app_code_from_conf_not_exists_type(path):
    try:
        get_app_code_from_conf(Path(path).stem, 'not_exists_conf_path', 'not_exists_type')
        pytest.fail('Not raised error')
    except GenerateCodeError as e:
        assert str(e) == 'Config type not_exists_type not supported.'


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
@pytest.mark.parametrize(
    'config_path, config_type',
    [('test_json_config', 'json'), ('test_yaml_config', 'yaml')],
    ids=['json', 'yaml']
)
@pytest.mark.parametrize('with_custom_conf', [False, True])
def test_get_app_code_from_conf_success(path, config_path, config_type, with_custom_conf):
    if with_custom_conf:
        kwargs_str = ''
        for k, v in CONF_KWARGS_DICT.items():
            kwargs_str += f"\nup.conf.{k} = {v}"

        generated_code = get_app_code_from_conf(path, config_path, config_type, **CONF_KWARGS_DICT)
        expected_code = GET_APP_CODE_FROM_CONF_TEMPLATE.format(
            path=Path(path).stem,
            config_type=config_type,
            config_path=config_path,
            custom_conf=kwargs_str,
        )
    else:
        generated_code = get_app_code_from_conf(path, config_path, config_type)
        expected_code = GET_APP_CODE_FROM_CONF_TEMPLATE.format(
            path=Path(path).stem,
            config_type=config_type,
            config_path=config_path,
            custom_conf='',
        )

    assert generated_code == expected_code


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
def test_get_app_code_from_up_bin_not_exists_file(path):
    assert get_app_code_from_up_bin(path, 'not_exist_path', 'pickle') == GET_APP_CODE_FROM_PICKLE_TEMPLATE.format(
        path=Path(path).stem,
        binary_path='not_exist_path',
        custom_conf='',
    )


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
def test_get_app_code_from_up_bin_not_exists_type(path):
    try:
        get_app_code_from_up_bin(Path(path).stem, 'not_exists_path', 'not_exists_type')
        pytest.fail('Not raised error')
    except GenerateCodeError as e:
        assert str(e) == 'Binary type not_exists_type not supported.'


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
@pytest.mark.parametrize(
    'template, binary_path, binary_type',
    [
        (GET_APP_CODE_FROM_PICKLE_TEMPLATE, 'binary.pckl', 'pickle'),
        (GET_APP_CODE_FROM_JOBLIB_TEMPLATE, 'binary.joblib', 'joblib')
    ],
    ids=['pickle', 'joblib']
)
@pytest.mark.parametrize('with_custom_conf', [False, True])
def test_get_app_code_from_up_bin_success(path, template, binary_path, binary_type, with_custom_conf):
    if with_custom_conf:
        kwargs_str = ''
        for k, v in CONF_KWARGS_DICT.items():
            kwargs_str += f"\nup.conf.{k} = {v}"

        generated_code = get_app_code_from_up_bin(path, binary_path, binary_type, **CONF_KWARGS_DICT)
        expected_code = template.format(
            path=Path(path).stem,
            binary_path=binary_path,
            custom_conf=kwargs_str,
        )
    else:
        generated_code = get_app_code_from_up_bin(path, binary_path, binary_type)
        expected_code = template.format(path=Path(path).stem, binary_path=binary_path, custom_conf='')

    assert generated_code == expected_code


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
def test_get_app_code_from_model_bin_not_exists_file(path):
    assert get_app_code_from_model_bin(path, 'not_exist_model.onnx') == GET_APP_CODE_FROM_MODEL_BIN_TEMPLATE.format(
        path=Path(path).stem,
        model_path='not_exist_model.onnx',
        model_name='not_exist_model.onnx',
        custom_conf='',
    )


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
@pytest.mark.parametrize('with_custom_conf', [False, True])
def test_get_app_code_from_model_bin_success(path, with_custom_conf, pickle_print_model):
    if with_custom_conf:
        kwargs_str = ''
        for k, v in CONF_KWARGS_DICT.items():
            kwargs_str += f"\n        {k}={v},"

        generated_code = get_app_code_from_model_bin(path, pickle_print_model, **CONF_KWARGS_DICT)
        expected_code = GET_APP_CODE_FROM_MODEL_BIN_TEMPLATE.format(
            path=Path(path).stem,
            model_path=pickle_print_model,
            model_name=Path(pickle_print_model).name,
            custom_conf=kwargs_str,
        )
    else:
        generated_code = get_app_code_from_model_bin(path, pickle_print_model)
        expected_code = GET_APP_CODE_FROM_MODEL_BIN_TEMPLATE.format(
            path=Path(path).stem,
            model_path=pickle_print_model,
            model_name=Path(pickle_print_model).name,
            custom_conf=''
        )

    assert generated_code == expected_code


@pytest.mark.parametrize('force', [False, True])
@pytest.mark.parametrize(
    'app_name, app_code',
    [(p, 'test app code') for p in PATCHES_TO_FILE],
    ids=PATCHES_TO_FILE
)
def test_save_app_exists_file(tmp_path_factory, app_name, app_code, force):
    path = tmp_path_factory.getbasetemp() / 'test_save_app_exists_file' / app_name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(app_code)

    try:
        save_app(str(path), app_code, force)
        if force is True:
            with open(path, 'r') as f:
                assert f.read() == app_code
        else:
            pytest.fail('Not raised error')
    except SaveAppError as e:
        if force is True:
            pytest.fail(f'Raise error {e}')
        else:
            assert str(e) == f'File "{path}" exists. For replace, use argument force=True'


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
def test_make_app_with_config_and_binary(path):
    try:
        make_app(
            path_to_file=path,
            use_config=True,
            use_binary=True,
        )
        pytest.fail('Not raised error')
    except InputArgsError as e:
        assert str(e) == 'You can specify only one of the arguments: use_config [-c], use_binary [-b], use_model [-m].'


@pytest.mark.parametrize('path', PATCHES_TO_FILE)
def test_make_app_without_config_and_binary(path):
    try:
        make_app(
            path_to_file=path,
            use_config=True,
            path_to_src=None,
        )
        pytest.fail('Not raised error')
    except InputArgsError as e:
        assert str(e) == 'For arguments use_config [-c], use_binary [-b], use_model [-m] need set argument path_to_src [-s].'


def test_make_app_with_not_exists_src_path():
    try:
        make_app(
            path_to_file='nox_exists_path',
            use_config=True,
            path_to_src='not_exists_src_path',
        )
    except FileNotFoundError as e:
        assert str(e) == 'File "not_exists_src_path" not exists.'


def test_make_app_with_exists_path(tmp_path_factory):
    path = tmp_path_factory.getbasetemp() / 'test_make_app_with_exists_path' / \
           'test_run_with_not_exists_path.py'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('')

    try:
        make_app(path_to_file=str(path))
    except FileExistsError as e:
        assert str(e) == f'File "{path}" exists.'


def test_make_app_change_extension(tmp_path_factory):
    path = tmp_path_factory.getbasetemp() / 'test_make_app_change_extension' / 'test_make_app_change_extension'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    make_app(path_to_file=str(path))
    assert os.path.exists(str(path) + '.py')
    assert not os.path.exists(path)

    path = tmp_path_factory.getbasetemp() / 'test_make_app_change_extension' / 'test_make_app_change_extension_2'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    make_app(path_to_file=str(path), extension='.exe')
    assert os.path.exists(str(path) + '.exe')
    assert not os.path.exists(path)

    path = tmp_path_factory.getbasetemp() / 'test_make_app_change_extension' / 'test_make_app_change_extension_3.py'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    make_app(path_to_file=str(path))
    assert os.path.exists(str(path))
