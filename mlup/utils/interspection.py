import inspect
import logging
from typing import Callable

from mlup.constants import IS_X


logger = logging.getLogger('MLup')


def analyze_method_params(func: Callable, auto_detect_predict_params: bool = True, ignore_self: bool = True):
    """
    Parse method and create dict with method parameters.

    :param Callable func: Func for parse arguments.
    :param bool auto_detect_predict_params: Auto detect predict params and set List to first param.
    :param bool ignore_self: Ignore first param with name self and cls.

    :return: Dict with parsing func arguments.
    :rtype: Dict[str, Dict[str, str]]

    def example(a, b = 100, *, c: float = 123):
        pass

    method_params = analyze_method_params(method)
    method_params
    [{'name': 'a', 'required': True},
     {'name': 'b', 'required': False, 'default': 100},
     {'name': 'c', 'type': float, 'required': False, 'default': 123}]

    """
    sign = inspect.signature(func)
    result = []
    types = {
        int: 'int',
        float: 'float',
        bool: 'bool',
        str: 'str',
    }
    logger.info(f'Auto analyzing arguments in {func}.')

    _found_X = False
    for i, (param_name, param_obj) in enumerate(sign.parameters.items()):
        # Skip first arg if it's self, cls
        if i == 0 and ignore_self:
            if param_name.lower() in {'self', 'cls'}:
                continue

        param_data = {
            'name': param_name,
            'required': True,
        }
        if param_obj.annotation is not inspect._empty:
            if param_obj.annotation in types:
                param_data['type'] = types[param_obj.annotation]
            else:
                logger.warning(
                    f'For model predict argument writes not supported type {param_obj.annotation}. '
                    f'Skip added validation'
                )

        if param_obj.default is not inspect._empty:
            param_data['required'] = False
            param_data['default'] = param_obj.default
            if 'type' not in param_data:
                param_data['type'] = types[type(param_obj.default)]

        if param_name.lower().strip() == 'x' and auto_detect_predict_params:
            logger.info(f'Found X param in model params. Set List type')
            param_data['type'] = 'List'
            param_data[IS_X] = True
            _found_X = True

        result.append(param_data)

    if not _found_X and auto_detect_predict_params:
        if result:
            logger.info(
                f'X argument in predict method not found. '
                f'For predict data use first argument with name {result[0]["name"]}.'
            )
            result[0]['type'] = 'List'
            result[0][IS_X] = True
        else:
            logger.info('Not found arguments in predict method.')

    return result

