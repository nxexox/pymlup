import json
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
import datetime
from time import sleep
from typing import Dict, Optional, List

import pytest
import requests
from websocket import create_connection


logger = logging.getLogger('mlup.test')


class JupyterCellCallError(Exception):
    def __init__(self, content, msg):
        self.content = content
        self.msg = msg

    def __str__(self):
        return f'Content={self.content}, msg={self.msg}'


@dataclass
class JupyterNotebookTestServer:
    ip: str = '0.0.0.0'
    port: int = 8888
    protocol: str = 'http'
    token = 'mytoken123456'
    notebooks_folder: str = ''

    _server_start_command_tmp: str = "jupyter notebook --ip {ip} --port {port} --NotebookApp.token='{token}' " \
                                     "--no-browse {notebooks_folder}"
    _server_stop_command_tmp: str = "jupyter notebook stop {port}"
    _start_proc: subprocess.Popen = field(init=False, repr=False)
    _stop_proc: subprocess.Popen = field(init=False, repr=False)

    @property
    def headers(self) -> Dict:
        return {"Authorization": f"Token {self.token}"}

    @property
    def server_host(self) -> str:
        return f'{self.protocol}://{self.ip}:{self.port}'

    def start_server(self):
        self._start_proc = subprocess.Popen(
            self._server_start_command_tmp.format(
                ip=self.ip, port=self.port, token=self.token, notebooks_folder=self.notebooks_folder,
            ).split()
        )
        logger.info('Waiting start jupyter notebooks server 5 seconds')
        sleep(5)

    def copy_notebook_to_work_folder(self, notebook_name: str):
        pass

    def stop_server(self) -> Optional[str]:
        if self._start_proc is None:
            return None
        self._stop_proc = subprocess.Popen(
            self._server_stop_command_tmp.format(port=self.port).split()
        )
        output, error = self._stop_proc.communicate()
        if error:
            raise ValueError(str(error))
        self._stop_proc = None
        self._stop_proc = None
        return output

    def send_execute_request(self, code):
        msg_type = "execute_request"
        content = {"code": code, "silent": False}
        hdr = {
            "msg_id": uuid.uuid1().hex,
            "username": "test",
            "session": uuid.uuid1().hex,
            "data": datetime.datetime.now().isoformat(),
            "msg_type": msg_type,
            "version": "5.0",
        }
        msg = {"header": hdr, "parent_header": hdr, "metadata": {}, "content": content}
        return msg

    def parse_notebook_results(self, results: List[Dict]):
        res = []
        for r in results:
            logger.info(f'Result call cell {r}')
            _d = r['data']
            if 'text/plain' in _d:
                res.append(_d['text/plain'])
            else:
                res.append(_d)
        return res

    def run_notebook(self, notebook_path: str):
        url = f"{self.server_host}/api/kernels"
        with requests.post(url, headers=self.headers) as response:
            kernel = json.loads(response.text)

        # Load the notebook and get the code of each cell
        url = f"{self.server_host}/api/contents/{notebook_path}"
        with requests.get(url, headers=self.headers) as response:
            file = json.loads(response.text)

        # filter out non-code cells like markdown
        code = [
            c["source"]
            for c in file["content"]["cells"]
            if len(c["source"]) > 0 and c["cell_type"] == "code"
        ]

        # Execution request/reply is done on websockets channels
        ws = create_connection(
            f"{'ws' if self.protocol == 'http' else 'wss'}://{self.ip}:{self.port}/api/kernels/{kernel['id']}/channels",
            header=self.headers,
        )

        for c in code:
            ws.send(json.dumps(self.send_execute_request(c)))

        code_blocks_to_execute = len(code)

        results = []

        while code_blocks_to_execute > 0:
            try:
                rsp = json.loads(ws.recv())
                msg_type = rsp["msg_type"]
                if msg_type == "error":
                    raise JupyterCellCallError(content=rsp["content"], msg=rsp["content"]["traceback"][0])
            except Exception as _e:
                print(f'ERROR: {_e}')
                break

            if msg_type == "execute_result":
                results.append(rsp["content"])

            if (
                msg_type == "execute_reply"
                and rsp["metadata"].get("status") == "ok"
                and rsp["metadata"].get("dependencies_met", False)
            ):
                code_blocks_to_execute -= 1

        ws.close()

        # Delete the kernel
        url = f"{self.server_host}/api/kernels/{kernel['id']}"
        requests.delete(url, headers=self.headers)
        return self.parse_notebook_results(results)


@pytest.fixture(scope="session")
def jupyter_notebook_server(root_dir):
    server = JupyterNotebookTestServer(notebooks_folder=root_dir)
    try:
        server.start_server()
        yield server
    finally:
        server.stop_server()
