import inspect
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

import yaml

from mlup.interfaces import MLupModelInterface
from mlup.web_app.app import MLupWebApp


logger = logging.getLogger('MLup')
KEYS_FOR_COLLAPSE = {
    "model": "mlup_model",
    "web_app": "web_app",
}
EXCLUDE_KEYS = {
    "model": {},
    "web_app": {"mlup_model",}
}


@dataclass
class MLupConfig:
    mlup_model: MLupModelInterface
    web_app: MLupWebApp = None

    def get_param_names(self, obj):
        sign = inspect.signature(obj.__init__)
        return {param_name for param_name in sign.parameters.keys() if param_name.lower() != 'self'}

    def set_params_to_obj(self, obj, conf: Dict):
        for k, v in conf.items():
            setattr(obj, k, v)

    def load_from_json(self, file_path: str):
        logger.info(f'Load config from {file_path}')
        with open(file_path, 'r') as f:
            _conf = json.load(f)
            for key, cls_name in KEYS_FOR_COLLAPSE.items():
                obj = getattr(self, cls_name)
                if obj is not None:
                    logger.debug(f'Set params to {obj.__class__}')
                    self.set_params_to_obj(obj, _conf.get(key, {}))

    def load_from_yaml(self, file_path: str):
        logger.info(f'Load config from {file_path}')
        with open(file_path, 'r') as f:
            _conf = yaml.safe_load(f)
            for key, cls_name in KEYS_FOR_COLLAPSE.items():
                obj = getattr(self, cls_name)
                if obj is not None:
                    logger.debug(f'Set params to {obj.__class__}')
                    self.set_params_to_obj(obj, _conf.get(key, {}))

    def save_to_json(self, file_path: str):
        _conf = {}
        logger.info(f'Save config to {file_path}')
        for key, cls_name in KEYS_FOR_COLLAPSE.items():
            _conf.setdefault(key, {})
            obj = getattr(self, cls_name)
            for param_name in self.get_param_names(obj):
                if param_name in EXCLUDE_KEYS.get(cls_name, {}):
                    continue
                val = getattr(obj, param_name)
                if isinstance(val, Enum):
                    val = val.value
                _conf[key][param_name] = val

        with open(file_path, 'w') as f:
            json.dump(_conf, f)

    def save_to_yaml(self, file_path: str):
        _conf = {}
        logger.info(f'Save config to {file_path}')
        for key, cls_name in KEYS_FOR_COLLAPSE.items():
            _conf.setdefault(key, {})
            obj = getattr(self, cls_name)
            for param_name in self.get_param_names(obj):
                if param_name in EXCLUDE_KEYS.get(cls_name, {}):
                    continue
                val = getattr(obj, param_name)
                if isinstance(val, Enum):
                    val = val.value
                _conf[key][param_name] = val

        with open(file_path, 'w') as f:
            yaml.dump(_conf, f)


LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "logging.Formatter",
            "fmt": "[%(process)d][%(thread)d] [%(asctime)s.%(msecs)03d] %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        "MLup": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}
