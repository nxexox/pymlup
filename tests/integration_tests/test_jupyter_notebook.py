import logging

import pytest


logger = logging.getLogger("mlup.test")


@pytest.mark.parametrize(
    'test_case',
    [
        'run_easy.ipynb',
        'run_worker_and_queue.ipynb',
        'run_batching.ipynb'
    ],
)
def test_run_web_app(jupyter_notebook_runner, tests_jupyter_notebooks_datadir, test_case):
    res = jupyter_notebook_runner.run_notebook(tests_jupyter_notebooks_datadir, test_case)
    # First item is up.predict result, second item is requests.post result
    assert res == ['[1, 2, 3]', "{'predict_result': [1, 2, 3]}"]


def test_call_predict_from(jupyter_notebook_runner, tests_jupyter_notebooks_datadir):
    res = jupyter_notebook_runner.run_notebook(tests_jupyter_notebooks_datadir, 'call_predict_from.ipynb')
    # First item is up.predict result, second item is requests.post result
    assert res == ['array([1, 2, 3])']
