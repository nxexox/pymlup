import importlib
import inspect
import logging
from enum import Enum
from typing import Callable, Any, Union, Optional, Type

from mlup.constants import IS_X, THERE_IS_ARGS, DEFAULT_X_ARG_NAME, BinarizationType, LoadedFile
from mlup.utils.profiling import TimeProfiler

logger = logging.getLogger('mlup')


def get_class_by_path(path_to_class: Union[str, Enum]) -> Any:
    """
    Get class by path to class. Use importlib.import_module.

    :param Union[str, Enum] path_to_class: Path to class for import and return.

    """
    if isinstance(path_to_class, Enum):
        path_to_class = path_to_class.value

    # Try get class by path
    try:
        return import_obj(path_to_class)
    except (ModuleNotFoundError, AttributeError) as e:
        logger.warning(f'Object {path_to_class} import error. {e}')
        raise


def import_obj(path_to_import_obj: str) -> Any:
    """
    Import any python object.

    :param str path_to_import_obj: Path for import object.

    :return: Imported object.

    """
    mod = importlib.import_module(path_to_import_obj.rsplit('.', 1)[0])
    return getattr(mod, path_to_import_obj.rsplit('.', 1)[1])


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
    arg_spec = inspect.getfullargspec(func)
    result = []
    types = {
        int: 'int',
        float: 'float',
        bool: 'bool',
        str: 'str',
    }
    is_there_args = False
    logger.info(f'Analyzing arguments in {func}.')

    _found_X = False
    for i, (param_name, param_obj) in enumerate(sign.parameters.items()):
        if arg_spec.varargs == param_name:
            is_there_args = True
        if arg_spec.varargs == param_name or arg_spec.varkw == param_name:
            logger.warning(f'Found unpacked argument: {arg_spec}. Skip argument')
            continue

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
            if 'type' not in param_data and type(param_obj.default) in types:
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
                f'For predict data use first argument with name "{result[0]["name"]}".'
            )
            result[0]['type'] = 'List'
            result[0][IS_X] = True
        else:
            logger.info('Not found arguments in predict method.')

    if not result and is_there_args:
        logger.info(
            f'Found not naming args in predict method. For call predict, use "{DEFAULT_X_ARG_NAME}" X key. '
            f'Example: mlup_model.predict({DEFAULT_X_ARG_NAME}=[obj_for_pred_1, obj_for_pred_2, ...]).'
        )
        result.append({'name': DEFAULT_X_ARG_NAME, 'required': False, THERE_IS_ARGS: True})
    return result


def auto_search_binarization_type(loaded_file: LoadedFile) -> Optional[Type[BinarizationType]]:
    """
    Search binarizer by model raw binary data and model name.
    Search in only mlup binarizers from mlup.constants.BinarizationType.

    :param LoadedFile loaded_file: Model raw data information.

    :return: Found binarizer class if found or None if not found.
    :rtype: Optional[Type[BinarizationType]]

    """
    logger.info(f'Run auto search binarizer.')
    probabilities = []
    with TimeProfiler('Time to auto search binarizer:', log_level='info'):
        for binarizer_path in BinarizationType:
            try:
                binarization_class = get_class_by_path(binarizer_path.value)
            except (ModuleNotFoundError, AttributeError):
                continue

            probability = binarization_class.is_this_type(loaded_file=loaded_file)
            probabilities.append((probability, binarizer_path))
            logger.debug(f'Binarizer "{binarizer_path}" have probability {probability}.')

        high_prob = sorted(probabilities, key=lambda x: x[0])[-1]
        logger.debug(f'Hypothesis binarizer "{high_prob[1]}" with probaility {high_prob[0]}.')
        if high_prob[0] > 0.0:
            logger.info(f'Found binarizer "{high_prob[1]}" with probaility {high_prob[0]}.')
            return high_prob[1]

    logger.info('Binarizer not found.')
    return None
