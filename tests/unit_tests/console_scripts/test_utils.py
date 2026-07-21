import argparse
from typing import Dict, List, Optional, Union

import pytest

from mlup.constants import StorageType, WebAppArchitecture
from mlup.console_scripts.utils import get_config_parser, get_set_fields, make_up_value_converter


# --- make_up_value_converter: unit tests for each Config field type ---

def test_convert_str_is_passthrough():
    convert = make_up_value_converter('predict_method_name', str)
    assert convert('__call__') == '__call__'
    assert convert('') == ''


@pytest.mark.parametrize('raw, expected', [('0', 0), ('8123', 8123), ('-5', -5)])
def test_convert_int_valid(raw, expected):
    convert = make_up_value_converter('port', int)
    assert convert(raw) == expected
    assert isinstance(convert(raw), int)


@pytest.mark.parametrize('raw', ['abc', '', '1.5', 'None'])
def test_convert_int_invalid(raw):
    convert = make_up_value_converter('port', int)
    with pytest.raises(argparse.ArgumentTypeError):
        convert(raw)


@pytest.mark.parametrize('raw, expected', [('1.0', 1.0), ('0.5', 0.5), ('10', 10.0), ('-2.5', -2.5)])
def test_convert_float_valid(raw, expected):
    convert = make_up_value_converter('batch_worker_timeout', float)
    assert convert(raw) == expected
    assert isinstance(convert(raw), float)


@pytest.mark.parametrize('raw', ['abc', '', 'yes'])
def test_convert_float_invalid(raw):
    convert = make_up_value_converter('batch_worker_timeout', float)
    with pytest.raises(argparse.ArgumentTypeError):
        convert(raw)


@pytest.mark.parametrize('raw', ['true', 'True', 'TRUE', '1', 'yes', 'YES', 'on', 'On'])
def test_convert_bool_true_values(raw):
    convert = make_up_value_converter('use_thread_loop', bool)
    assert convert(raw) is True


@pytest.mark.parametrize('raw', ['false', 'False', 'FALSE', '0', 'no', 'NO', 'off', 'Off'])
def test_convert_bool_false_values(raw):
    convert = make_up_value_converter('use_thread_loop', bool)
    assert convert(raw) is False


@pytest.mark.parametrize('raw', ['maybe', '2', '', 'truee'])
def test_convert_bool_invalid(raw):
    convert = make_up_value_converter('use_thread_loop', bool)
    with pytest.raises(argparse.ArgumentTypeError):
        convert(raw)


def test_convert_list_valid_json():
    convert = make_up_value_converter('columns', Optional[List[Dict[str, str]]])
    assert convert('["a", "b"]') == ['a', 'b']
    assert convert('[{"name": "col", "type": "list"}]') == [{'name': 'col', 'type': 'list'}]


def test_convert_list_invalid_json():
    convert = make_up_value_converter('columns', Optional[List[Dict[str, str]]])
    with pytest.raises(argparse.ArgumentTypeError):
        convert('not a json list')


def test_convert_dict_valid_json():
    convert = make_up_value_converter('uvicorn_kwargs', Dict)
    assert convert('{"workers": 1}') == {'workers': 1}


def test_convert_dict_invalid_json():
    convert = make_up_value_converter('uvicorn_kwargs', Dict)
    with pytest.raises(argparse.ArgumentTypeError):
        convert('{not valid json}')


@pytest.mark.parametrize('raw', ['null', 'None', 'NULL', 'none', 'NoNe'])
def test_convert_optional_int_null_values(raw):
    convert = make_up_value_converter('max_thread_loop_workers', Optional[int])
    assert convert(raw) is None


def test_convert_optional_int_real_value():
    convert = make_up_value_converter('max_thread_loop_workers', Optional[int])
    assert convert('4') == 4


@pytest.mark.parametrize('raw', ['null', 'none'])
def test_convert_optional_str_null_values(raw):
    convert = make_up_value_converter('dtype_for_predict', Optional[str])
    assert convert(raw) is None


def test_convert_optional_str_real_value():
    convert = make_up_value_converter('dtype_for_predict', Optional[str])
    assert convert('float32') == 'float32'


def test_convert_null_not_supported_for_non_optional_field():
    # port is a plain int (not Optional), so 'null' must fail like any other non-numeric string,
    # not silently become None.
    convert = make_up_value_converter('port', int)
    with pytest.raises(argparse.ArgumentTypeError):
        convert('null')


def test_convert_ambiguous_union_enum_like_field_is_passthrough():
    # storage_type/binarization_type/data_transformer_for_* are Union[str, SomeEnum]: these are
    # dotted import-path strings resolved via reflection elsewhere, so they must stay untouched.
    convert = make_up_value_converter('storage_type', Union[str, StorageType])
    assert convert('mlup.ml.storage.local_disk.DiskStorage') == 'mlup.ml.storage.local_disk.DiskStorage'
    assert convert('some.custom.path.MyStorage') == 'some.custom.path.MyStorage'


def test_convert_plain_enum_field_is_passthrough():
    # mode is typed as a plain WebAppArchitecture (not a Union), but is still accepted as a string.
    convert = make_up_value_converter('mode', WebAppArchitecture)
    assert convert('mlup.web.architecture.batching.BatchingSingleProcessArchitecture') == \
        'mlup.web.architecture.batching.BatchingSingleProcessArchitecture'


# --- get_config_parser / get_set_fields: end-to-end argparse wiring ---

def _parse_up_args(monkeypatch, argv):
    monkeypatch.setattr('sys.argv', ['mlup', 'run', 'model.pckl', '-m'] + argv)
    parser = get_config_parser(argparse.ArgumentParser('mlup run'))
    return get_set_fields(parser)


def test_get_set_fields_only_returns_explicitly_set_fields(monkeypatch):
    result = _parse_up_args(monkeypatch, ['--up.port=8123'])
    assert result == {'port': 8123}


def test_get_set_fields_converts_port_to_int(monkeypatch):
    result = _parse_up_args(monkeypatch, ['--up.port=8123'])
    assert result['port'] == 8123
    assert isinstance(result['port'], int)


def test_get_set_fields_converts_max_thread_loop_workers_to_int(monkeypatch):
    result = _parse_up_args(monkeypatch, ['--up.max_thread_loop_workers=4'])
    assert result['max_thread_loop_workers'] == 4
    assert isinstance(result['max_thread_loop_workers'], int)


def test_get_set_fields_converts_use_thread_loop_to_bool_false(monkeypatch):
    result = _parse_up_args(monkeypatch, ['--up.use_thread_loop=False'])
    assert result['use_thread_loop'] is False


def test_get_set_fields_converts_columns_to_list(monkeypatch):
    result = _parse_up_args(monkeypatch, ['--up.columns=["a", "b"]'])
    assert result['columns'] == ['a', 'b']


def test_get_set_fields_converts_uvicorn_kwargs_to_dict(monkeypatch):
    result = _parse_up_args(monkeypatch, ['--up.uvicorn_kwargs={"workers": 1}'])
    assert result['uvicorn_kwargs'] == {'workers': 1}


def test_get_set_fields_bad_int_exits_cleanly_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['mlup', 'run', 'model.pckl', '-m', '--up.max_thread_loop_workers=abc'])
    parser = get_config_parser(argparse.ArgumentParser('mlup run'))
    with pytest.raises(SystemExit) as exc_info:
        get_set_fields(parser)
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert 'Traceback' not in captured.err
    assert "expected an integer, got 'abc'" in captured.err
