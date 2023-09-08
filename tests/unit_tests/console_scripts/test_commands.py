import pytest

from mlup.console_scripts.command import run_command


@pytest.mark.parametrize(
    'bash_string',
    ['', ' ', '   ', '              ']
)
def test_run_command_short_command(bash_string):
    try:
        run_command([bash_string])
        pytest.fail('Not raised error')
    except SystemExit as e:
        assert e.code == 1


@pytest.mark.parametrize(
    'bash_string',
    ['mlup -h', 'mlup --help', 'mlup help', 'mlup  -h', 'mlup    help', 'mlup -h --help', 'mlup -h --help help']
)
def test_run_command_help_command(bash_string):
    try:
        run_command(bash_string.split())
        pytest.fail('Not raised error')
    except SystemExit as e:
        assert e.code == 1


@pytest.mark.parametrize(
    'bash_string',
    ['mlup not_exists_command', 'mlup not-exists-command']
)
def test_run_command_not_exists_module(bash_string):
    try:
        run_command(bash_string.split())
        pytest.fail('Not raised error')
    except SystemExit as e:
        assert e.code == 1


@pytest.mark.parametrize(
    'bash_string',
    ['validate_config', 'validate-config']
)
def test_run_command_exists_module(bash_string):
    try:
        run_command(bash_string.split())
        pytest.fail('Not raised error')
    except SystemExit as e:
        # Need additional args for command
        assert e.code == 2
