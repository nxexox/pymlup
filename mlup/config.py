import inspect
import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Set, Union

import yaml

from mlup.constants import StorageType


def set_logging_settings(logging_config: Dict):
    logging.config.dictConfig(logging_config)


LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "mlup": {
            "()": "mlup.utils.logging.SwitchFormatter",
            "fmt": "%(levelname)s:[%(asctime)s.%(msecs)03d] - %(message)s",
            "fmt_web": "%(levelprefix)s [%(process)d][%(thread)d] [%(asctime)s.%(msecs)03d] - %(message)s",
            "fmt_console_commands": "%(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "uvicorn": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(message)s",
            "use_colors": None,
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s [%(process)d][%(thread)d] [%(asctime)s.%(msecs)03d] - %(client_addr)s "%(request_line)s" %(status_code)s',  # noqa: E501
        },
    },
    "handlers": {
        "mlup": {
            "formatter": "mlup",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "uvicorn": {
            "formatter": "uvicorn",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "mlup": {"handlers": ["mlup"], "level": "INFO", "propagate": False},
        "uvicorn": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
    },
}

set_logging_settings(LOGGING_CONFIG)
logger = logging.getLogger('mlup')
CONFIGS_VERSION = '1'
CONFIGS = {"ml", "web"}
EXCLUDE_KEYS = {
    "ml": {"ml", "_data_transformer_for_predict", "_data_transformer_for_predicted"},
    "web": {"ml", "custom_column_pydantic_model"}
}
EXCLUDE_CONDITION_KEYS = [
    {'condition': {'storage_type': StorageType.memory.value}, 'skip': ['storage_kwargs']}
]


@dataclass
class ConfigProvider:
    obj_for_config: Any
    configs_objects_map: Dict

    def _get_param_names(self, obj: Any) -> Set:
        sign = inspect.signature(obj.__init__)
        return {param_name for param_name in sign.parameters.keys() if param_name.lower() != 'self'}

    def _set_params_to_obj(self, obj, conf: Dict, exclude_keys: Set):
        for k, v in conf.items():
            if k not in exclude_keys:
                setattr(obj.conf, k, v)

    def set_config_from_dict(self, config: Dict):
        for cls_name in CONFIGS:
            obj = getattr(self.obj_for_config, cls_name)
            if obj is not None:
                logger.debug(f'Set params to {obj.__class__}')
                self._set_params_to_obj(obj, config.get(cls_name, {}), EXCLUDE_KEYS.get(cls_name, {}))

    def get_config_dict(self) -> Dict:
        config = {}
        for cls_name in CONFIGS:
            config.setdefault(cls_name, {})
            obj = getattr(self.obj_for_config, cls_name)

            for param_name in self._get_param_names(self.configs_objects_map[cls_name]):
                if param_name in EXCLUDE_KEYS.get(cls_name, {}):
                    continue
                try:
                    val = getattr(obj.conf, param_name)
                except AttributeError as e:
                    raise e
                if isinstance(val, Enum):
                    val = val.value
                config[cls_name][param_name] = val

            for exclude in EXCLUDE_CONDITION_KEYS:
                if all(_k in config[cls_name] and config[cls_name][_k] == _v for _k, _v in exclude['condition'].items()):
                    for skip in exclude['skip']:
                        config[cls_name].pop(skip, None)

        return config

    def load_from_dict(self, conf: Dict):
        logger.info(f'Load config from dict')
        self.set_config_from_dict(conf)

    def load_from_json(self, file_path: Union[str, Path]):
        logger.info(f'Load config from {file_path}')
        with open(file_path, 'r') as f:
            _conf = json.load(f)
            self.set_config_from_dict(_conf)

    def load_from_yaml(self, file_path: Union[str, Path]):
        logger.info(f'Load config from {file_path}')
        with open(file_path, 'r') as f:
            _conf = yaml.safe_load(f)
            self.set_config_from_dict(_conf)

    def save_to_json(self, file_path: Union[str, Path]):
        logger.info(f'Save config to {file_path}')
        config_dict = self.get_config_dict()
        config_dict['version'] = CONFIGS_VERSION
        with open(file_path, 'w') as f:
            json.dump(config_dict, f)

    def save_to_yaml(self, file_path: Union[str, Path]):
        logger.info(f'Save config to {file_path}')
        config_dict = self.get_config_dict()
        with open(file_path, 'w') as f:
            f.write(f"version: '{CONFIGS_VERSION}'\n")
            yaml.dump(config_dict, f)
