import argparse
import logging
import os
import sys

import mlup
from mlup.utils.logging import configure_logging_formatter


configure_logging_formatter('console_commands')
logger = logging.getLogger('mlup')


class ValidationConfigError(Exception):
    pass


def validate_config(config_path: str, config_type: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'File "{config_path}" not found.')

    if config_type == 'json':
        mlup.UP.load_from_json(conf_path=config_path, load_model=False)
    elif config_type == 'yaml':
        mlup.UP.load_from_yaml(conf_path=config_path, load_model=False)
    else:
        raise ValidationConfigError(f'Config type {config_type} not supported.')


def main():
    parser = argparse.ArgumentParser(
        'mlup validate-config',
        usage='%(prog)s path\nFor more information use "%(prog)s --help"',
        description="Command for validate your config file on semantic correct.\n"
                    "This only runs the code depending on your arguments:\n"
                    "  --type=json - mlup.UP.load_from_json(json_conf_path, load_model=False)\n"
                    "  --type=yaml - mlup.UP.load_from_yaml(yaml_conf_path, load_model=False)",
        epilog="Examples:\n"
               "  mlup validate-config /path/to/my/config/file.yaml\n"
               "  mlup validate-config --type=json /path/to/my/config/file.json",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Add hide argument for command name, because command run from two commands: mlup <command_name>.
    parser.add_argument('command_name', type=str, help=argparse.SUPPRESS)
    parser.add_argument('path', type=str, help='Path to file with config.')
    parser.add_argument(
        '--type', nargs='?', default='yaml', const='yaml', choices=['json', 'yaml'],
        help='Type config file. Can use only: yaml, json. Default is yaml.'
    )

    args, unknown = parser.parse_known_args()

    try:
        validate_config(args.path, args.type)
        logger.info('Config is valid')
    except ValidationConfigError as e:
        logger.error(str(e))
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f'Config not is valid: "{e}"')
        sys.exit(1)
