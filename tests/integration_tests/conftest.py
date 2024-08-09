import logging
import os
from typing import List

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
import pytest


logger = logging.getLogger("mlup.test")


class JupyterNotebookRunner:
    @staticmethod
    def parse_results(nb: nbformat.NotebookNode, output_type="execute_result") -> List[str]:
        """
        Parse the output of the jupyter notebook run.

        :param nbformat.NotebookNode nb: Notebook for parse.
        :param str output_type: Type of result output.
            Available: execute_result, stream.

        :return: Result of the parsed output.

        """
        result = []
        for c in nb.cells:
            if c.cell_type == 'code':
                for o in c.outputs:
                    if o.output_type == output_type:
                        if o.output_type == "execute_result":
                            result.append(o.data["text/plain"])
                        elif o.output_type == "stream":
                            result.append(o.text)
        return result

    def run_notebook(self, notebook_path: str, notebook_name: str) -> List[str]:
        nb = nbformat.read(os.path.join(notebook_path, notebook_name), as_version=4)
        client = NotebookClient(nb, timeout=600)
        try:
            client.execute()
        except CellExecutionError as e:
            logger.exception(e)
            raise e

        return self.parse_results(nb, output_type="execute_result")


@pytest.fixture(scope="session")
def jupyter_notebook_runner():
    yield JupyterNotebookRunner()
