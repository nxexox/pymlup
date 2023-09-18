import json
import logging
from dataclasses import fields, dataclass, make_dataclass, field

import pytest
import yaml

from mlup.config import ConfigProvider, EXCLUDE_KEYS, EXCLUDE_CONDITION_KEYS, CONFIGS_VERSION
from mlup.constants import ModelLibraryType, StorageType, ModelDataTransformerType, ITEM_ID_COL_NAME, \
    WebAppArchitecture
from mlup.up import Config
from mlup.ml.model import ModelConfig
from mlup.web.app import WebAppConfig


logger = logging.getLogger('mlup.test')


_test_config = {
    'name': 'MyFirstMLupModel',
    'version': '1.0.0.0',
    'type': 'ModelLibraryType.SKLEARN',
    'columns': None,
    'custom_column_pydantic_model': None,
    'predict_method_name': 'predict',
    'auto_detect_predict_params': True,
    'storage_type': 'StorageType.memory',
    'storage_kwargs': {},
    'binarization_type': 'BinarizationType.PICKLE',
    'use_thread_loop': True,
    'max_thread_loop_workers': None,
    'data_transformer_for_predict': 'ModelDataTransformerType.PANDAS_DF',
    'data_transformer_for_predicted': 'ModelDataTransformerType.NUMPY_ARR',
    'dtype_for_predict': None,
    'host': '0.0.0.0',
    'port': 8009,
    'web_app_version': '1.0.0.0',
    'column_validation': False,
    'mode': 'WebAppArchitecture.directly_to_predict',
    'max_queue_size': 100,
    'ttl_predicted_data': 60,
    'ttl_client_wait': 30.0,
    'min_batch_len': 10,
    'batch_worker_timeout': 1.0,
    'is_long_predict': False,
    'throttling_max_requests': None,
    'throttling_max_request_len': None,
    'timeout_for_shutdown_daemon': 3.0,
    'item_id_col_name': 'ITEM_ID_COL_NAME',
    'debug': False,
    'show_docs': True,
    'uvicorn_kwargs': {},
}


def test_up_config_default_values():
    conf = Config()
    # Model configs
    assert conf.name == 'MyFirstMLupModel'
    assert conf.version == '1.0.0.0'
    assert conf.type == ModelLibraryType.SKLEARN
    assert conf.columns is None
    assert conf.predict_method_name == 'predict'
    assert conf.auto_detect_predict_params is True
    assert conf.storage_type == StorageType.memory
    assert conf.storage_kwargs == {}
    assert conf.binarization_type == 'auto'
    assert conf.use_thread_loop is True
    assert conf.max_thread_loop_workers is None
    assert conf.data_transformer_for_predict == ModelDataTransformerType.NUMPY_ARR
    assert conf.data_transformer_for_predicted == ModelDataTransformerType.NUMPY_ARR
    assert conf.dtype_for_predict is None

    # WebAPp configs
    assert conf.host == '0.0.0.0'
    assert conf.port == 8009
    assert conf.web_app_version == '1.0.0.0'
    assert conf.column_validation is False
    assert conf.custom_column_pydantic_model is None
    assert conf.mode == WebAppArchitecture.directly_to_predict
    assert conf.max_queue_size == 100
    assert conf.ttl_predicted_data == 60
    assert conf.ttl_client_wait == 30.0
    assert conf.min_batch_len == 10
    assert conf.batch_worker_timeout == 1.0
    assert conf.is_long_predict is False
    assert conf.throttling_max_requests is None
    assert conf.throttling_max_request_len is None
    assert conf.timeout_for_shutdown_daemon == 3.0
    assert conf.item_id_col_name == ITEM_ID_COL_NAME
    assert conf.debug is False
    assert conf.show_docs is True
    assert conf.uvicorn_kwargs == {}

    config_actual_fields = set()
    for n, f in Config.__dataclass_fields__.items():
        if f.repr is True:
            config_actual_fields.add(n)

    assert config_actual_fields - _test_config.keys() == set()


def test_config_have_unique_keys():
    model_conf_fields = {f.name for f in fields(ModelConfig)}
    web_app_conf_fields = {f.name for f in fields(WebAppConfig)}
    assert model_conf_fields & web_app_conf_fields == set()


@dataclass
class _Conf:
    pass


@dataclass
class ForExample:
    @dataclass
    class _ML:
        conf = _Conf()

    @dataclass
    class _WEB:
        conf = _Conf()

    ml: _ML = field(default_factory=_ML)
    web: _WEB = field(default_factory=_WEB)


class TestConfigProvider:
    def make_for_example_with_new_config(self, new_config: _Conf):
        @dataclass
        class NewForExample:
            @dataclass
            class _ML:
                conf = new_config

            @dataclass
            class _WEB:
                conf = new_config
            ml: _ML = field(default_factory=_ML)
            web: _WEB = field(default_factory=_WEB)
        return NewForExample()

    def test_get_config_dict(self):
        obj = ForExample()
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        conf_dict = provider.get_config_dict()
        assert conf_dict == {'ml': {}, 'web': {}}

    def test_get_config_dict_with_data(self):
        @dataclass
        class _NewConf(_Conf):
            without_default_int: int
            default_string: str = 'default_string'

        obj = self.make_for_example_with_new_config(_NewConf(123))
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        config_dict = provider.get_config_dict()
        assert config_dict == {
            'ml': {'without_default_int': 123, 'default_string': 'default_string'},
            'web': {'without_default_int': 123, 'default_string': 'default_string'}
        }

    def test_get_config_dict_with_exclude_keys(self):
        _NewConf = make_dataclass(
            '_NewConf',
            [
                (attr_name, str, field(default=f'default_{attr_name}'))
                for attr_name in set([a for b in EXCLUDE_KEYS.values() for a in b])
            ]
        )

        obj = self.make_for_example_with_new_config(_NewConf())
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        config_dict = provider.get_config_dict()

        ml_keys = EXCLUDE_KEYS['web'] - EXCLUDE_KEYS['ml']
        web_app_keys = EXCLUDE_KEYS['ml'] - EXCLUDE_KEYS['web']

        assert config_dict == {
            'ml': {k: f'default_{k}' for k in ml_keys},
            'web': {k: f'default_{k}' for k in web_app_keys}
        }

    def test_get_config_dict_with_condition_exclude_keys(self):
        _fields = []
        _skip_fields = set()
        random_field = None
        random_skip = None
        for cond in EXCLUDE_CONDITION_KEYS:
            for k, v in cond['condition'].items():
                _fields.append((k, type(v), v))
                if not random_field:
                    random_field = k
            if not random_skip:
                random_skip = cond['skip']
            for skip in cond['skip']:
                _fields.append((skip, str, f'default_{skip}'))
                _skip_fields.add(skip)
        _fields.append(('TEST', str, 'default_TEST'))

        _NewConf = make_dataclass('_NewConf', _fields)

        obj = self.make_for_example_with_new_config(_NewConf())
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        config_dict = provider.get_config_dict()
        assert config_dict == {
            'ml': {k: v for k, t, v in _fields if k not in _skip_fields},
            'web': {k: v for k, t, v in _fields if k not in _skip_fields}
        }

        # Check change condition on valid value
        obj = self.make_for_example_with_new_config(_NewConf(**{random_field: 'NewValue'}))
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        config_dict = provider.get_config_dict()
        _skip_fields = _skip_fields - set(random_skip)
        assert config_dict == {
            'ml': {k: v if k != random_field else 'NewValue' for k, t, v in _fields if k not in _skip_fields},
            'web': {k: v if k != random_field else 'NewValue' for k, t, v in _fields if k not in _skip_fields}
        }

    def test_set_config_from_dict(self):
        obj = ForExample()
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        config_dict = {
            'ml': {
                'test': 'test',
                'two_test': 123,
            },
            'web': {
                'three_test': 431,
                'bool_test': True,
            }
        }
        provider.set_config_from_dict(config_dict)
        for k, v in config_dict['ml'].items():
            assert getattr(obj.ml.conf, k) == v
        for k, v in config_dict['web'].items():
            assert getattr(obj.web.conf, k) == v

    def test_set_config_from_dict_with_exclude(self):
        obj = ForExample()
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        config_dict = {
            'ml': {
                'test': 'test',
                'two_test': 123,
                **{k: f'default_{k}' for k in EXCLUDE_KEYS['ml']}
            },
            'web': {
                'three_test': 431,
                'bool_test': True,
                **{k: f'default_{k}' for k in EXCLUDE_KEYS['web']}
            }
        }
        provider.set_config_from_dict(config_dict)
        for k, v in config_dict['ml'].items():
            if k not in EXCLUDE_KEYS['ml']:
                assert getattr(obj.ml.conf, k) == v
            else:
                assert hasattr(obj.ml.conf, k) is False
        for k, v in config_dict['web'].items():
            if k not in EXCLUDE_KEYS['web']:
                assert getattr(obj.web.conf, k) == v
            else:
                assert hasattr(obj.web.conf, k) is False

    @pytest.mark.parametrize(
        'file_name, load_func',
        [
            ('test_save_to_json.json', json.load),
            ('test_save_to_yaml.yaml', yaml.safe_load),
        ],
        ids=['to_json', 'to_yaml']
    )
    def test_save_to(self, tmp_path_factory, file_name, load_func):
        @dataclass
        class _NewConf(_Conf):
            without_default_int: int
            default_string: str = 'default_string'

        obj = self.make_for_example_with_new_config(_NewConf(123))
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        provider.save_to_json(tmp_path_factory.getbasetemp() / file_name)

        with open(tmp_path_factory.getbasetemp() / file_name, 'rb') as f:
            data = load_func(f)

        assert data == {
            'ml': {'without_default_int': 123, 'default_string': 'default_string'},
            'web': {'without_default_int': 123, 'default_string': 'default_string'},
            'version': CONFIGS_VERSION,
        }

    @pytest.mark.parametrize(
        'file_name, load_func',
        [
            ('test_save_to_json_with_exclude_keys.json', json.load),
            ('test_save_to_yaml_with_exclude_keys.yaml', yaml.safe_load),
        ],
        ids=['to_json', 'to_yaml']
    )
    def test_save_to_with_exclude_keys(self, tmp_path_factory, file_name, load_func):
        _NewConf = make_dataclass(
            '_NewConf',
            [
                (attr_name, str, field(default=f'default_{attr_name}'))
                for attr_name in set([a for b in EXCLUDE_KEYS.values() for a in b])
            ]
        )

        obj = self.make_for_example_with_new_config(_NewConf())
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        provider.save_to_json(tmp_path_factory.getbasetemp() / file_name)

        with open(tmp_path_factory.getbasetemp() / file_name, 'rb') as f:
            data = load_func(f)

        ml_keys = EXCLUDE_KEYS['web'] - EXCLUDE_KEYS['ml']
        web_app_keys = EXCLUDE_KEYS['ml'] - EXCLUDE_KEYS['web']

        assert data == {
            'ml': {k: f'default_{k}' for k in ml_keys},
            'web': {k: f'default_{k}' for k in web_app_keys},
            'version': CONFIGS_VERSION,
        }

    @pytest.mark.parametrize(
        'file_name, save_func_name, load_func_name',
        [
            ('test_load_from_json.json', 'save_to_json', 'load_from_json'),
            ('test_load_from_yaml.yaml', 'save_to_yaml', 'load_from_yaml'),
        ],
        ids=['from_json', 'from_yaml']
    )
    def test_load_from(self, tmp_path_factory, file_name, save_func_name, load_func_name):
        @dataclass
        class _NewConf(_Conf):
            without_default_int: int
            default_string: str = 'default_string'

        obj = self.make_for_example_with_new_config(_NewConf(123))
        provider = ConfigProvider(obj, configs_objects_map={"ml": obj._ML.conf, "web": obj._WEB.conf})
        getattr(provider, save_func_name)(tmp_path_factory.getbasetemp() / file_name)

        new_obj = self.make_for_example_with_new_config(_NewConf(321, 'new_string'))
        new_provider = ConfigProvider(
            new_obj,
            configs_objects_map={"ml": new_obj._ML.conf, "web": new_obj._WEB.conf}
        )
        assert new_provider.get_config_dict() == {
            'ml': {'without_default_int': 321, 'default_string': 'new_string'},
            'web': {'without_default_int': 321, 'default_string': 'new_string'}
        }

        getattr(new_provider, load_func_name)(tmp_path_factory.getbasetemp() / file_name)
        assert new_provider.get_config_dict() == {
            'ml': {'without_default_int': 123, 'default_string': 'default_string'},
            'web': {'without_default_int': 123, 'default_string': 'default_string'}
        }
