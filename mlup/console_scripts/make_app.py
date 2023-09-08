import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict

from mlup.console_scripts.utils import get_config_parser, get_set_fields, CHANGE_SETTINGS_HELP_TEXT
from mlup.utils.logging import configure_logging_formatter


configure_logging_formatter('console_commands')
logger = logging.getLogger('mlup')


app_template = """{import_additional_library}


{code_by_load_src}{conf_kwargs}{code_by_load_app}
up.web.load()

# If you want to run the application yourself, or add something else to it, use this variable.
# Example with uvicorn: uvicorn {file_name}:app --host 0.0.0.0 --port 80
app = up.web.app

if __name__ == '__main__':
    up.run_web_app()
"""

free_app_load_template = """# You can load the model yourself and pass it to the "ml_model" argument.
# up = mlup.UP(ml_model=my_model, conf=mlup.Config())
up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.
        # You can use storage_type and storage_kwargs for auto_load model from storage.{conf_kwargs}
    )
)
"""

from_model_app_load_template = """up = mlup.UP(
    conf=mlup.Config(
        # Set your config, for work model and get model.{conf_kwargs}
    )
)
"""

pickle_load_template = """with open('{path_to_bin_file}', 'rb') as f:
    up = pickle.load(f)
"""


class MakeAppError(Exception):
    pass


class InputArgsError(MakeAppError):
    pass


class GenerateCodeError(MakeAppError):
    pass


class SaveAppError(MakeAppError):
    pass


def _conf_kwargs_to_str(
    config_line_prefix: str = '',
    kv_sep: str = '=',
    end_sep: str = ',',
    **config_kwargs
) -> str:
    kwargs_str = ''
    for k, v in config_kwargs.items():
        kwargs_str += f"\n{config_line_prefix}{k}{kv_sep}{v}{end_sep}"
    return kwargs_str


def get_app_code_without_preload(path_to_app, **config_kwargs):
    return app_template.format(
        file_name=Path(path_to_app).stem,
        import_additional_library='import mlup',
        code_by_load_src=free_app_load_template.format(
            conf_kwargs=_conf_kwargs_to_str('        ', **config_kwargs),
        ),
        code_by_load_app='up.ml.load()',
        conf_kwargs='',
    )


def get_app_code_from_conf(path_to_app: str, config_path: str, config_type: str, **config_kwargs) -> str:
    if config_type == 'json':
        return app_template.format(
            file_name=Path(path_to_app).stem,
            import_additional_library='import mlup',
            code_by_load_src=f"up = mlup.UP.load_from_json('{config_path}', load_model=False)",
            conf_kwargs=_conf_kwargs_to_str('up.conf.', kv_sep=' = ', end_sep='', **config_kwargs),
            code_by_load_app='\nup.ml.load()',
        )
    elif config_type == 'yaml':
        return app_template.format(
            file_name=Path(path_to_app).stem,
            import_additional_library='import mlup',
            code_by_load_src=f"up = mlup.UP.load_from_yaml('{config_path}', load_model=False)",
            conf_kwargs=_conf_kwargs_to_str('up.conf.', kv_sep=' = ', end_sep='', **config_kwargs),
            code_by_load_app='\nup.ml.load()',
        )
    raise GenerateCodeError(f'Config type {config_type} not supported.')


def get_app_code_from_up_bin(path_to_app: str, binary_path: str, binary_type: str, **config_kwargs) -> str:
    if binary_type == 'pickle':
        return app_template.format(
            file_name=Path(path_to_app).stem,
            import_additional_library='import pickle',
            code_by_load_src=pickle_load_template.format(path_to_bin_file=binary_path),
            conf_kwargs=_conf_kwargs_to_str('up.conf.', kv_sep=' = ', end_sep='', **config_kwargs),
            code_by_load_app='\nif not up.ml.loaded:\n    up.ml.load()',
        )
    elif binary_type == 'joblib':
        return app_template.format(
            file_name=Path(path_to_app).stem,
            import_additional_library='import joblib',
            code_by_load_src=f"up = joblib.load('{binary_path}')\n",
            conf_kwargs=_conf_kwargs_to_str('up.conf.', kv_sep=' = ', end_sep='', **config_kwargs),
            code_by_load_app='\nif not up.ml.loaded:\n    up.ml.load()',
        )
    raise GenerateCodeError(f'Binary type {binary_type} not supported.')


def get_app_code_from_model_bin(path_to_app: str, model_path: str, **model_config_kwargs):
    cp = '        '  # config line prefix
    kwargs_str = _conf_kwargs_to_str('        ', **model_config_kwargs)
    return app_template.format(
        file_name=Path(path_to_app).stem,
        import_additional_library='import mlup\nfrom mlup import constants',
        code_by_load_src=from_model_app_load_template.format(
            conf_kwargs=f"\n{cp}storage_type=constants.StorageType.disk,"
                        f"\n{cp}storage_kwargs=" + "{"
                        f"\n{cp}    'path_to_files': '{model_path}',"
                        f"\n{cp}    'files_mask': '{Path(model_path).name}',"
                        f"\n{cp}" + '},'
                        f"{kwargs_str}"
        ),
        code_by_load_app='up.ml.load()',
        conf_kwargs='',
    )


def save_app(path_to_app: str, app_code: str, force: bool = False):
    if not force:
        if os.path.exists(path_to_app):
            raise SaveAppError(f'File "{path_to_app}" exists. For replace, use argument force=True')

    folder = os.path.dirname(path_to_app)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path_to_app, 'w') as f:
        f.write(app_code)


def make_app(
    path_to_file: str,
    path_to_src: Optional[str] = None,
    use_config: bool = False,
    use_binary: bool = False,
    use_model: bool = False,
    config_type: str = 'yaml',
    binary_type: str = 'pickle',
    force: bool = False,
    extension: str = '.py',
    config_fields: Optional[Dict] = None,
):
    if any(((use_config and use_binary), (use_config and use_model), (use_binary and use_model))):
        raise InputArgsError(
            'You can specify only one of the arguments: use_config [-c], use_binary [-b], use_model [-m].'
        )

    if (use_config or use_binary or use_model) and not path_to_src:
        raise InputArgsError(
            'For arguments use_config [-c], use_binary [-b], use_model [-m] need set argument path_to_src [-s].'
        )

    if path_to_src and not os.path.exists(path_to_src):
        raise FileNotFoundError(f'File "{path_to_src}" not exists.')

    if os.path.isfile(path_to_file) and not force:
        raise FileExistsError(f'File "{path_to_file}" exists.')

    if not path_to_file.endswith(extension):
        path_to_file += extension

    if use_config:
        code = get_app_code_from_conf(
            path_to_app=path_to_file,
            config_path=path_to_src,
            config_type=config_type,
            **(config_fields or {})
        )
    elif use_binary:
        code = get_app_code_from_up_bin(
            path_to_app=path_to_file,
            binary_path=path_to_src,
            binary_type=binary_type,
            **(config_fields or {})
        )
    elif use_model:
        code = get_app_code_from_model_bin(
            path_to_app=path_to_file,
            model_path=path_to_src,
            **(config_fields or {})
        )
    else:
        code = get_app_code_without_preload(path_to_app=path_to_file)

    try:
        save_app(path_to_app=path_to_file, app_code=code, force=force)
        logger.info(f'App success created: {os.path.abspath(path_to_file)}')
    except SaveAppError:
        logger.error(f'App not created: {path_to_file}')
        raise


def main():
    parser = argparse.ArgumentParser(
        'mlup make-app',
        usage='%(prog)s path_to_app\nFor more information use "%(prog)s --help"',
        description="Command for make .py file with web_app and your conf or model.\n"
                    "You can specify one of the arguments:\n"
                    "  -m - for make app with load from binary model file.\n"
                    "  -c - for make app with load from conf file.\n"
                    "  -b - for make app with load from mlup.UP binary object.\n"
                    "OR don't say anything, and than app will be with empty settings.",
        epilog=f'{CHANGE_SETTINGS_HELP_TEXT}\n'
               'After generate python web app file, you can run with python command or uvicorn command:\n'
               '    python /path/to/generated/web_app.py\n'
               '    cd /path/to/generated && uvicorn your_file_name:app --host 0.0.0.0 --port 8009\n'
               '\nExamples:\n'
               '  Make from model:\n'
               '    mlup make-app -ms ~/my_model.onnx ~/web_app.py -f  # With default settings\n'
               '    mlup make-app -ms ~/my_model.onnx ~/web_app.py --up.port=8001 '
               '--up.predict_method_name=\\"__call__\\" --up.use_thread_loop=True '
               '--up.columns=\'[{"name": "col", "type": "list"}]\'  # With custom settings\n'
               '\n  Make from config:\n'
               '    mlup make-app -cs ~/mlup-conf.yaml ~/web_app.py\n'
               '    mlup make-app -cs ~/mlup-conf.json -ct json ~/web_app.py\n'
               '    mlup make-app -cs ~/mlup-conf.yaml ~/web_app.py --up.port=8001 '
               '--up.predict_method_name=\\"__call__\\" --up.use_thread_loop=True '
               '--up.columns=\'[{"name": "col", "type": "list"}]\'  # Add custom settings\n'
               '\n  Make from binary mlup.UP object:\n'
               '    mlup make-app -bs ~/mlup.pkl ~/web_app.py\n'
               '    mlup make-app -bs ~/mlup.joblib -bt joblib ~/web_app.py'
               '    mlup make-app -bs ~/mlup.pkl ~/web_app.py --up.port=8001 '
               '--up.predict_method_name=\\"__call__\\" --up.use_thread_loop=True '
               '--up.columns=\'[{"name": "col", "type": "list"}]\'  # Add custom settings',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Add hide argument for command name, because command run from two commands: mlup <command_name>.
    parser.add_argument('command_name', type=str, help=argparse.SUPPRESS)
    parser.add_argument('path_to_app', type=str, help='Path to result file with web app.')
    parser.add_argument(
        '-m', '--model', action='store_true',
        help='Make app from binary file with only model.'
    )
    parser.add_argument(
        '-c', '--conf', action='store_true',
        help='Make app from config file. Configuration file type can be specified with --conf-type.'
    )
    parser.add_argument(
        '-b', '--bin', action='store_true',
        help='Make app from binary file with mlup.UP object and model. '
             'Binarization type can be specified with argument --bin-type.'
    )
    parser.add_argument(
        '-s', '--src', type=str,
        help='Path to config or binary file with mlup.UP object or model. Use for make preload code.'
    )
    parser.add_argument('-f', '--force', action='store_true', help='Replace file by path, if --src file exists.')
    parser.add_argument(
        '-ct', '--conf-type', nargs='?', default='yaml', const='yaml', choices=['json', 'yaml'],
        help='Type config file. Can use only: yaml, json. Default is yaml.'
    )
    parser.add_argument(
        '-bt', '--bin-type', nargs='?', default='pickle', const='pickle', choices=['pickle', 'joblib'],
        help='Type binarization your mlup.UP. Can use only: pickle, joblib. Default is pickle.'
    )

    # Add mlup Config params, for run from model [-m]
    patched_parser = get_config_parser(parser)
    args, unknown = patched_parser.parse_known_args()

    try:
        make_app(
            path_to_file=args.path_to_app,
            path_to_src=args.src,
            use_config=args.conf,
            use_binary=args.bin,
            use_model=args.model,
            config_type=args.conf_type,
            binary_type=args.bin_type,
            force=args.force,
            config_fields=get_set_fields(patched_parser)
        )
    except (InputArgsError, FileExistsError, FileNotFoundError) as e:
        logger.error(str(e))
        sys.exit(1)
    except MakeAppError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f'Error with create app: "{e}"')
        sys.exit(1)
