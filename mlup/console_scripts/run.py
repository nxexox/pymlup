import argparse
import logging
import os
import pickle
import sys
from typing import Dict

import mlup
from mlup.console_scripts.utils import get_config_parser, get_set_fields, CHANGE_SETTINGS_HELP_TEXT
from mlup.constants import StorageType


logger = logging.getLogger('mlup')


class InputArgsError(Exception):
    pass


def run_from_config(config_path: str, config_type: str):
    if config_type == 'json':
        up = mlup.UP.load_from_json(conf_path=config_path, load_model=True)
        up.run_web_app()
    elif config_type == 'yaml':
        up = mlup.UP.load_from_yaml(conf_path=config_path, load_model=True)
        up.run_web_app()
    else:
        raise ValueError(f'Config type {config_type} not supported.')


def run_from_up_bin(binary_path: str, binary_type: str):
    if binary_type == 'pickle':
        with open(binary_path, 'rb') as f:
            up = pickle.load(f)
        up.run_web_app()
    elif binary_type == 'joblib':
        import joblib
        up = joblib.load(binary_path)
        up.run_web_app()
    else:
        raise ValueError(f'Binary type {binary_type} not supported.')


def run_from_model(model_path: str, **kwargs):
    if 'binarization_type' not in kwargs:
        kwargs['binarization_type'] = 'auto'
    up = mlup.UP(
        conf=mlup.Config(
            storage_type=StorageType.disk,
            storage_kwargs={
                'path_to_files': model_path,
                'files_mask': r'.+',
            },
            **kwargs,
        )
    )
    up.ml.load()
    up.run_web_app()


def run(
    path: str,
    conf_type: str = 'yaml',
    bin_type: str = 'pickle',
    use_model: bool = False,
    use_conf: bool = False,
    use_bin: bool = False,
    config_fields: Dict = None,
    verbose: bool = False,
):
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug('Run in verbose mode')

    if any(((use_conf and use_bin), (use_conf and use_model), (use_bin and use_model))):
        raise InputArgsError('You can specify only one of the arguments: use_conf [-c], use_bin [-b], use_model [-m].')
    if not use_conf and not use_bin and not use_model:
        raise InputArgsError('You must specify one of the arguments: use_conf [-c], use_bin [-b], use_model [-m].')

    if not os.path.exists(path):
        raise FileNotFoundError(f'Path {path} not exists.')

    if use_conf:
        run_from_config(path, conf_type)
    elif use_bin:
        run_from_up_bin(path, bin_type)
    elif use_model:
        run_from_model(path, **config_fields)
    else:
        raise ValueError(
            f"Something went wrong.\n"
            f"path='{path}', conf_type='{conf_type}', bin_type='{bin_type}', use_conf={use_conf}, use_bin={use_bin}."
        )


def main():
    parser = argparse.ArgumentParser(
        'mlup run',
        usage='%(prog)s path\nFor more information use "%(prog)s --help"',
        description="Command for run web application with uvicorn python interface.\n"
                    "You must specify one of the arguments:\n"
                    "  -m - for run web app from binary model file.\n"
                    "  -c - for run web app from config file.\n"
                    "  -b - for run web app with load from mlup.UP binary object.\n\n"
                    "You can set custom uvicorn settings in conf.uvicorn_kwargs dictionary or --up.uvicorn_kwargs. "
                    "See examples.",
        epilog=f'{CHANGE_SETTINGS_HELP_TEXT}\n'
               'Examples:\n'
               '  Run from model:\n'
               '    mlup run -m ~/my_model.onnx  # With default settings\n'
               '    mlup run -m ~/mu_model.onnx --up.port=8011 '
               '--up.binarization_type=\\"mlup.ml.binarization.onnx.InferenceSessionBinarizer\\" '
               '--up.uvicorn_kwargs=\'{"workers": 4, "timeout_graceful_shutdown": 10}\' '
               '--up.use_thread_loop=True --up.columns=\'[{"name": "col", "type": "list"}]\'  # Add custom settings\n'
               '\n  Run from config:\n'
               '    mlup run -c ~/mlup-conf.yaml\n'
               '    mlup run -c ~/mlup-conf.json -ct json\n'
               '    mlup run -c ~/mlup-conf.yaml '
               '--up.uvicorn_kwargs=\'{"workers": 4, "timeout_graceful_shutdown": 10}\'  # Add custom settings\n'
               '\n  Run from binary mlup.UP object:\n'
               '    mlup run -b ~/mlup.pkl\n'
               '    mlup run -b ~/mlup.joblib -bt joblib\n'
               '    mlpu run -b ~/mlup.pkl '
               '--up.uvicorn_kwargs=\'{"workers": 4, "timeout_graceful_shutdown": 10}\'  # Add custom settings',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Add hide argument for command name, because command run from two commands: mlup <command_name>.
    parser.add_argument('command_name', type=str, help=argparse.SUPPRESS)
    parser.add_argument('path', type=str, help='Path to config or binary file with mlup.UP object or model.')
    parser.add_argument('-m', '--model', action='store_true', help='Run web app from binary model file.')
    parser.add_argument(
        '-c', '--conf', action='store_true',
        help='Run web app from config file. Configuration file type can be specified with argument --conf-type.'
    )
    parser.add_argument(
        '-b', '--bin', action='store_true',
        help='Run web app from binary file with mlup.UP object. '
             'Binarization type can be specified with argument --bin-type.'
    )
    parser.add_argument(
        '-ct', '--conf-type', nargs='?', default='yaml', const='yaml', choices=['json', 'yaml'],
        help='Type config file. Can use only: yaml, json. Default is yaml.'
    )
    parser.add_argument(
        '-bt', '--bin-type', nargs='?', default='pickle', const='pickle', choices=['pickle', 'joblib'],
        help='Type binarization your mlup.UP object. Can use only: pickle, joblib. Default is pickle.'
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Set DEBUG logger level in application, for see more information about work application.\n'
             'Default logger level is INFO.'
    )

    # Add mlup Config params, for run from model [-m]
    patched_parser = get_config_parser(parser)
    args, unknown = patched_parser.parse_known_args()

    try:
        run(
            path=args.path,
            conf_type=args.conf_type,
            bin_type=args.bin_type,
            use_conf=args.conf,
            use_bin=args.bin,
            use_model=args.model,
            config_fields=get_set_fields(patched_parser),
            verbose=args.verbose,
        )
    except InputArgsError as e:
        parser.error(str(e))
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
