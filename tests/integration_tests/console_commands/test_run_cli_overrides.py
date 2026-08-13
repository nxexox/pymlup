"""
Integration tests for `mlup run -m <model> --up.<field>=<value>` CLI overrides.

These reproduce, end-to-end through the real CLI (argparse + subprocess), the five
override invocations flagged by docs/project-audit.md as broken before the
--up.<field> type-coercion fix in mlup/console_scripts/utils.py: every value used to
be passed to mlup.Config as a raw string, which silently misconfigured bool/int/dict
fields or crashed the process outright. These tests check the actual running
application's behavior (HTTP responses), not just the Config object.
"""
import logging
import subprocess
import sys
import time

import requests


logger = logging.getLogger('mlup.test')


def _run_mlup_cli(*extra_args: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, '-m', 'mlup.console_scripts.command', 'run', *extra_args],
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )


def _wait_for(url: str, method: str = 'get', json=None, attempts: int = 15):
    response, error = None, None
    for i in range(attempts):
        try:
            # A per-request timeout is required here: without one, a server that accepts the
            # TCP connection but never responds would hang this call (and the whole test run)
            # indefinitely, regardless of the bounded `attempts` retry count.
            response = requests.request(method, url, json=json, timeout=5)
        except (requests.ConnectionError, requests.Timeout) as e:
            logger.error(f'Attempt number {i}. Request error {e}.')
            time.sleep(i)
        except Exception as e:
            error = e
            break
        else:
            break
    return response, error


def _stop(proc: subprocess.Popen) -> str:
    proc.kill()
    try:
        _, output = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.wait()
        output = ''
    return output or ''


def test_cli_override_port(scikit_learn_binary_cls_model):
    """--up.port=8123 (from the README's own examples) must not crash uvicorn's own
    startup log line and must actually bind the app on the requested port."""
    proc = _run_mlup_cli('-m', str(scikit_learn_binary_cls_model.path), '--up.port=8130')
    try:
        response, error = _wait_for('http://0.0.0.0:8130/health')
    finally:
        output = _stop(proc)

    if error:
        raise error
    assert response is not None
    assert response.status_code == 200
    assert response.json() == {'status': 200}
    assert 'Traceback' not in output
    assert 'Logging error' not in output


def test_cli_override_max_thread_loop_workers(scikit_learn_binary_cls_model):
    """--up.max_thread_loop_workers=4 used to crash inside ThreadPoolExecutor(...) because
    the value stayed the string '4'. It must now load, run, and actually be an int."""
    proc = _run_mlup_cli(
        '-m', str(scikit_learn_binary_cls_model.path),
        '--up.port=8131', '--up.max_thread_loop_workers=4', '--up.debug=True',
    )
    try:
        response, error = _wait_for('http://0.0.0.0:8131/info')
    finally:
        output = _stop(proc)

    if error:
        raise error
    assert response is not None
    assert response.status_code == 200
    body = response.json()
    assert body['model_config']['max_thread_loop_workers'] == 4
    assert 'Traceback' not in output


def test_cli_override_use_thread_loop_false(scikit_learn_binary_cls_model):
    """--up.use_thread_loop=False used to set the truthy string 'False', so thread-loop mode
    could never actually be disabled from the CLI. It must now be the real bool False, and
    predict must still work correctly in synchronous (non-threaded) mode."""
    proc = _run_mlup_cli(
        '-m', str(scikit_learn_binary_cls_model.path),
        '--up.port=8132', '--up.use_thread_loop=False', '--up.debug=True',
    )
    try:
        info_response, info_error = _wait_for('http://0.0.0.0:8132/info')
        predict_response, predict_error = _wait_for(
            'http://0.0.0.0:8132/predict',
            method='post',
            json={scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]},
        )
    finally:
        output = _stop(proc)

    if info_error:
        raise info_error
    if predict_error:
        raise predict_error

    assert info_response is not None
    body = info_response.json()
    assert body['model_config']['use_thread_loop'] is False

    assert predict_response is not None
    assert predict_response.status_code == 200
    assert predict_response.json() == {'predict_result': [scikit_learn_binary_cls_model.test_model_response_raw]}
    assert 'Traceback' not in output


def test_cli_override_columns(scikit_learn_binary_cls_model):
    """--up.columns='["a", "b"]' used to stay the raw JSON string instead of becoming a
    real list, silently misconfiguring the field. It must now be an actual Python list."""
    proc = _run_mlup_cli(
        '-m', str(scikit_learn_binary_cls_model.path),
        '--up.port=8133', '--up.columns=["a", "b"]',
    )
    try:
        response, error = _wait_for('http://0.0.0.0:8133/info')
    finally:
        output = _stop(proc)

    if error:
        raise error
    assert response is not None
    assert response.status_code == 200
    assert response.json()['model_info']['columns'] == ['a', 'b']
    assert 'Traceback' not in output


def test_cli_override_uvicorn_kwargs(scikit_learn_binary_cls_model):
    """--up.uvicorn_kwargs='{"workers": 1}' used to replace the whole dict with a raw JSON
    string, which crashed later when the app tried `conf.uvicorn_kwargs['host'] = ...`
    (TypeError: 'str' object does not support item assignment). It must now merge into a
    real dict and the app must still start correctly."""
    proc = _run_mlup_cli(
        '-m', str(scikit_learn_binary_cls_model.path),
        '--up.port=8134', '--up.uvicorn_kwargs={"workers": 1}',
    )
    try:
        response, error = _wait_for('http://0.0.0.0:8134/health')
    finally:
        output = _stop(proc)

    if error:
        raise error
    assert response is not None
    assert response.status_code == 200
    assert response.json() == {'status': 200}
    assert 'Traceback' not in output
