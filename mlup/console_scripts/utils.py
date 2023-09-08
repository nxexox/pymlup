import argparse
from dataclasses import asdict
from typing import Dict, Any

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

    for _f_name in asdict(mlup.Config()):
        if _f_name.startswith('_'):
            continue
        parser.add_argument('--up.' + _f_name, type=str, help=argparse.SUPPRESS, default=argparse.SUPPRESS)

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
