import importlib
import sys
from typing import List


AVAILABLE_COMMANDS = """Available commands:
  * validate-config     Validate your config for valid format for MLup.
  * run                 Run your mlup application, from different sources.
  * make-app            Create python file with mlup application code by your settings.
"""


HELP = f"""usage: mlup [-h] command

positional arguments:
  command               Command name for run

options:
  -h, --help            show this help message and exit
  
{AVAILABLE_COMMANDS}
"""


def run_command(args: List[str]):
    try:
        command, command_args = args[0], args[1:]
    except IndexError:
        print(HELP)
        sys.exit(1)

    if command.strip() in ('-h', '--help', 'help'):
        print(HELP)
        sys.exit()

    command = command.replace('-', '_')

    try:
        module = importlib.import_module(f'mlup.console_scripts.{command}')
    except ModuleNotFoundError as e:
        print(f'Invalid command {command} - mlup.console_scripts.{command}.')
        print(AVAILABLE_COMMANDS)
        sys.exit(1)

    getattr(module, 'main')()


def main():
    args = sys.argv[1:]
    run_command(args)


if __name__ == '__main__':
    main()
