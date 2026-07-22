import argparse
import json
from dataclasses import fields
from typing import Dict, Any, Optional, Tuple, Union, get_args, get_origin

import mlup


CHANGE_SETTINGS_HELP_TEXT = (
    'You can change any setting that is available in mlup.Config by specifying '
    '--up.{conf_variable_name}={your_value} of the setting.\n'
    'Examples for different data types:\n'
    '  --up.port=8011\n'
    '  --up.batch_worker_timeout=10.0\n'
    '  --up.predict_method_name=\\"__call__\\"\n'
    '  --up.use_thread_loop=False\n'
    '  --up.columns=\'[{"name": "col", "type": "list"}]\'\n'
    '  --up.uvicorn_kwargs=\'{"workers": 4, "timeout_graceful_shutdown": 10}\'\n'
)

_BOOL_TRUE_VALUES = {'true', '1', 'yes', 'on'}
_BOOL_FALSE_VALUES = {'false', '0', 'no', 'off'}
_NULL_VALUES = {'null', 'none'}


def _unwrap_optional(field_type: Any) -> Tuple[Optional[Any], bool]:
    """
    If field_type is Optional[T] (i.e. Union[T, None]), return (T, True).
    If field_type is an ambiguous Union with more than one non-None member
    (e.g. Union[str, SomeEnum], used for the storage_type/binarization_type/mode/... fields),
    return (None, is_optional) so the caller falls back to a plain passthrough.
    Otherwise return (field_type, False).
    """
    if get_origin(field_type) is Union:
        args = [a for a in get_args(field_type) if a is not type(None)]  # noqa: E721
        is_optional = type(None) in get_args(field_type)
        if len(args) == 1:
            return args[0], is_optional
        return None, is_optional
    return field_type, False


def make_up_value_converter(field_name: str, field_type: Any):
    """
    Build an argparse `type=` callable that converts a raw --up.<field_name> CLI string
    into the real type of the matching mlup.Config dataclass field, instead of always
    leaving it a str (which used to silently corrupt bool/int/float/list/dict settings).

    Conversion errors raise argparse.ArgumentTypeError, so argparse reports a clean
    "error: argument --up.<field>: ..." message and exits, instead of an unhandled traceback.
    """
    inner_type, is_optional = _unwrap_optional(field_type)
    target_type = inner_type if inner_type is not None else field_type

    def converter(raw: str):
        if is_optional and raw.strip().lower() in _NULL_VALUES:
            return None

        if target_type is bool:
            low = raw.strip().lower()
            if low in _BOOL_TRUE_VALUES:
                return True
            if low in _BOOL_FALSE_VALUES:
                return False
            raise argparse.ArgumentTypeError(
                f"expected a boolean (true/false/1/0/yes/no/on/off), got '{raw}'"
            )

        if target_type is int:
            try:
                return int(raw)
            except ValueError:
                raise argparse.ArgumentTypeError(f"expected an integer, got '{raw}'")

        if target_type is float:
            try:
                return float(raw)
            except ValueError:
                raise argparse.ArgumentTypeError(f"expected a float, got '{raw}'")

        if target_type is str:
            return raw

        if target_type in (list, dict) or get_origin(target_type) in (list, dict):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise argparse.ArgumentTypeError(f"expected valid JSON, got '{raw}' ({e})")

        # Enum-like/ambiguous typed fields (storage_type, binarization_type, mode, type,
        # data_transformer_for_*, custom_column_pydantic_model, ...) are passed through as-is:
        # mlup already accepts either a raw dotted-path/Enum-value string or an Enum instance
        # for these fields, so leaving them a string preserves existing, documented behavior.
        return raw

    return converter


def get_config_parser(parent_parser: argparse.ArgumentParser):
    parser = argparse.ArgumentParser(
        parent_parser.prog,
        usage=parent_parser.usage,
        description=parent_parser.description,
        epilog=parent_parser.epilog,
        formatter_class=parent_parser.formatter_class,
        parents=[parent_parser],
        add_help=False
    )

    for f in fields(mlup.Config):
        if f.name.startswith('_'):
            continue
        parser.add_argument(
            '--up.' + f.name,
            type=make_up_value_converter(f.name, f.type),
            help=argparse.SUPPRESS,
            default=argparse.SUPPRESS,
        )

    return parser


def get_set_fields(parser: argparse.ArgumentParser) -> Dict[str, Any]:
    args, _ = parser.parse_known_args()
    result = {}
    for arg_name, arg_value in args.__dict__.items():
        if not arg_name.startswith('up.'):
            continue
        arg_name = arg_name[3:]
        result[arg_name] = arg_value
    return result
