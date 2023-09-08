import asyncio
import logging
import time
from threading import Thread

import httpx
import pytest
import numpy

from mlup.config import ConfigProvider
from mlup.constants import ModelDataTransformerType
from mlup.ml.model import MLupModel
from mlup.up import UP, Config
from mlup.utils.loop import run_async
from mlup.web.app import MLupWebApp

logger = logging.getLogger('mlup.test')


class TestMLupPublicMethods:
    @staticmethod
    def assert_attributes(conf, obj):
        for key, value in conf.items():
            if isinstance(obj, dict):
                obj_val = obj[key]
            else:
                obj_val = getattr(obj, key)
            assert value == obj_val

    def test_create_mlup(self, print_model):
        up = UP(ml_model=print_model)

        assert isinstance(up.ml, MLupModel)
        assert isinstance(up.web, MLupWebApp)
        assert isinstance(up.config_provider, ConfigProvider)
        assert isinstance(up.conf, Config)

    def test_load_from_dict(self, test_dict_config):
        up = UP.load_from_dict(test_dict_config, load_model=False)

        self.assert_attributes(test_dict_config['ml'], up.ml.conf)
        self.assert_attributes(test_dict_config['web'], up.web.conf)

    def test_to_dict(self, test_dict_config):
        up = UP.load_from_dict(test_dict_config, load_model=False)
        dict_from_up = up.to_dict()

        self.assert_attributes(test_dict_config['ml'], dict_from_up['ml'])
        self.assert_attributes(test_dict_config['web'], dict_from_up['web'])

    def test_load_from_yaml(self, test_yaml_config, test_dict_config):
        up = UP.load_from_yaml(test_yaml_config, load_model=False)

        self.assert_attributes(test_dict_config['ml'], up.ml.conf)
        self.assert_attributes(test_dict_config['web'], up.web.conf)

    def test_to_yaml(self, test_yaml_config, test_dict_config):
        up = UP.load_from_yaml(test_yaml_config, load_model=False)
        up.to_yaml(str(test_yaml_config) + 'test')

        up = UP.load_from_yaml(str(test_yaml_config) + 'test', load_model=False)

        self.assert_attributes(test_dict_config['ml'], up.ml.conf)
        self.assert_attributes(test_dict_config['web'], up.web.conf)

    def test_load_from_json(self, test_json_config, test_dict_config):
        up = UP.load_from_yaml(test_json_config, load_model=False)

        self.assert_attributes(test_dict_config['ml'], up.ml.conf)
        self.assert_attributes(test_dict_config['web'], up.web.conf)

    def test_to_json(self, test_json_config, test_dict_config):
        up = UP.load_from_json(test_json_config, load_model=False)
        up.to_json(str(test_json_config) + 'test')

        up = UP.load_from_json(str(test_json_config) + 'test', load_model=False)

        self.assert_attributes(test_dict_config['ml'], up.ml.conf)
        self.assert_attributes(test_dict_config['web'], up.web.conf)

    @pytest.mark.asyncio
    async def test_predict(self, print_model):
        up = UP(
            ml_model=print_model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR
            )
        )
        up.ml.load()
        pred = up.predict(X=[[1, 2, 3]])
        assert pred == [[1, 2, 3]]

    @pytest.mark.asyncio
    async def test_async_predict(self, print_model):
        up = UP(
            ml_model=print_model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR
            )
        )
        up.ml.load()
        pred = await up.async_predict(X=[[1, 2, 3]])
        assert pred == [[1, 2, 3]]

    @pytest.mark.asyncio
    async def test_predict_from_numpy(self, print_model):
        up = UP(
            ml_model=print_model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
                data_transformer_for_predicted=ModelDataTransformerType.SRC_TYPES,
            )
        )
        up.ml.load()

        predicted_data = up.predict_from(X=[1, 2, 3, 4, 5, 6, 7, 8, 9])
        numpy.array_equal(predicted_data, numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))

    @pytest.mark.asyncio
    async def test_run_web_app_with_daemon_is_False(self, print_model):
        up = UP(
            ml_model=print_model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR
            )
        )
        up.ml.load()
        up.web.load()

        up.conf.uvicorn_kwargs['loop'] = 'none'

        web_app_thread = Thread(
            target=up.run_web_app,
            kwargs={'daemon': False},
            daemon=False,
            name='MLupWebAppDaemonTestsThread'
        )
        web_app_thread.start()
        await asyncio.sleep(0.5)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get('http://0.0.0.0:8009/health')
                assert response.status_code == 200
                assert response.json() == {'status': 200}
        finally:
            # Shutdown uvicorn not in main thread
            # https://stackoverflow.com/questions/58010119/are-there-any-better-ways-to-run-uvicorn-in-thread
            up.web._uvicorn_server.should_exit = True
            run_async(
                asyncio.wait_for,
                up.web._uvicorn_server.shutdown(),
                up.web.conf.timeout_for_shutdown_daemon
            )
            web_app_thread.join(timeout=3)

    @pytest.mark.asyncio
    async def test_run_with_daemon_is_True(self, print_model):
        up = UP(
            ml_model=print_model,
            conf=Config(
                port=8011,
                data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        up.ml.load()
        up.web.load()

        up.run_web_app(daemon=True)
        time_run = time.monotonic()
        while not up.web._uvicorn_server and time.monotonic() - time_run < 30.0:
            await asyncio.sleep(0.1)
        while up.web._uvicorn_server.started is False and time.monotonic() - time_run < 30.0:
            await asyncio.sleep(0.1)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get('http://0.0.0.0:8011/health')
                assert response.status_code == 200
                assert response.json() == {'status': 200}
        finally:
            up.stop_web_app(shutdown_timeout=3)
