import json

import pytest

from mlup.console_scripts.validate_config import validate_config, ValidationConfigError


@pytest.mark.parametrize(
    'config_path, config_type',
    [('not_exists_path.json', 'json'), ('not_exists_path.yaml', 'yaml')],
    ids=['json', 'yaml']
)
def test_validate_config_not_exists_conf(config_path, config_type):
    try:
        validate_config(config_path, config_type)
        pytest.fail('Not raised error')
    except FileNotFoundError as e:
        assert str(e) == f'File "{config_path}" not found.'


def test_validate_config_not_exists_conf_type(tmp_path_factory):
    file_path = tmp_path_factory.getbasetemp() / 'test_validate_config_not_exists_conf_type.json'
    with open(file_path, 'w') as f:
        f.write('Not valid config\n\nBut is valid multirows string.')

    try:
        validate_config(str(file_path), 'not_exists_type')
        pytest.fail('Not raised error')
    except ValidationConfigError as e:
        assert str(e) == 'Config type not_exists_type not supported.'


@pytest.mark.parametrize('config_type', ['json', 'yaml'])
def test_validate_config_not_valid_config(tmp_path_factory, config_type):
    file_path = tmp_path_factory.getbasetemp() / f'test_validate_config_not_valid_config.{config_type}'
    with open(file_path, 'w') as f:
        f.write('Not valid config\n\nBut is valid multirows string.')

    try:
        validate_config(str(file_path), config_type)
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
def test_validate_config_valid_config(request, config_fixture_name, conf_type):
    config_path = request.getfixturevalue(config_fixture_name)
    validate_config(config_path, conf_type)
