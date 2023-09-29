import asyncio
from threading import Thread
from typing import Dict
from unittest import TestCase

import httpx
import pytest
from fastapi import FastAPI

from mlup.constants import ModelDataTransformerType, ITEM_ID_COL_NAME, WebAppArchitecture
from mlup.errors import WebAppLoadError
from mlup.ml.model import MLupModel, ModelConfig
from mlup.utils.loop import run_async, create_async_task
from mlup.web.app import MLupWebApp, WebAppConfig


assertDictEqual = TestCase().assertDictEqual


DEFAULT_API_INFO_METHOD_DATA = {
    'model_info': {
        'name': 'MyFirstMLupModel',
        'version': '1.0.0.0',
        'type': 'sklearn',
        'columns': None
    },
    'web_app_info': {
        'version': '1.0.0.0',
    },
}


DEFAULT_API_DEBUG_INFO_METHOD_DATA = {
    **DEFAULT_API_INFO_METHOD_DATA,
    'web_app_config': {
        'host': 'localhost',
        'port': 8009,
        'web_app_version': '1.0.0.0',
        'column_validation': False,
        'custom_column_pydantic_model': None,
        'mode': 'mlup.web.architecture.directly_to_predict.DirectlyToPredictArchitecture',
        'max_queue_size': 100,
        'ttl_predicted_data': 60,
        'ttl_client_wait': 30.0,
        'min_batch_len': 10,
        'batch_worker_timeout': 1.0,
        'is_long_predict': False,
        'show_docs': True,
        'debug': True,
        'throttling_max_requests': None,
        'throttling_max_request_len': None,
        'timeout_for_shutdown_daemon': 3.0,
        'item_id_col_name': 'mlup_item_id'
    },
    'model_config': {
        'name': 'MyFirstMLupModel',
        'version': '1.0.0.0',
        'type': 'sklearn',
        'columns': None,
        'predict_method_name': 'predict',
        'auto_detect_predict_params': True,
        'storage_type': 'mlup.ml.storage.memory.MemoryStorage',
        'binarization_type': 'auto',
        'use_thread_loop': True,
        'max_thread_loop_workers': None,
        'data_transformer_for_predict': 'mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer',
        'data_transformer_for_predicted': 'mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer',
        'dtype_for_predict': None,
    }
}


def test_create_web_app(print_model):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(ml=mlup_model)

    assert mlup_web_app.conf.host == '0.0.0.0'
    assert mlup_web_app.conf.port == 8009
    assert mlup_web_app.conf.web_app_version == '1.0.0.0'
    assert mlup_web_app.conf.column_validation is False
    assert mlup_web_app.conf.custom_column_pydantic_model is None
    assert mlup_web_app.conf.mode is WebAppArchitecture.directly_to_predict
    assert mlup_web_app.conf.max_queue_size == 100
    assert mlup_web_app.conf.ttl_predicted_data == 60
    assert mlup_web_app.conf.ttl_client_wait == 30.0
    assert mlup_web_app.conf.min_batch_len == 10
    assert mlup_web_app.conf.batch_worker_timeout == 1.0
    assert mlup_web_app.conf.is_long_predict is False
    assert mlup_web_app.conf.throttling_max_requests is None
    assert mlup_web_app.conf.throttling_max_request_len is None
    assert mlup_web_app.conf.timeout_for_shutdown_daemon == 3.0
    assert mlup_web_app.conf.item_id_col_name == ITEM_ID_COL_NAME


def test_web_app_attribute(print_model):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(ml=mlup_model)
    mlup_model.load()

    try:
        mlup_web_app.app
        pytest.fail('Not raised error')
    except WebAppLoadError as e:
        assert str(e) == 'web nor creating. Please call web.load().'

    mlup_web_app.load()
    assert isinstance(mlup_web_app.app, FastAPI)


@pytest.mark.parametrize(
    'params, needed_api_points',
    [
        (
            {'is_long_predict': False, 'mode': WebAppArchitecture.directly_to_predict},
            {'/predict': {'POST', }, '/info': {'GET', }}
        ),
        (
            {'is_long_predict': True, 'mode': WebAppArchitecture.worker_and_queue},
            {'/predict': {'POST', }, '/info': {'GET', }, '/get-predict/{predict_id}': {'GET', }}
        ),
    ],
    ids=['is_long_predict=False', 'is_long_predict=True']
)
def test_load_web_app_check_api_points(print_model, params, needed_api_points):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(**params)
    )
    mlup_model.load()
    mlup_web_app.load()
    assert isinstance(mlup_web_app.app, FastAPI)

    for api_point in mlup_web_app.app.routes:
        if api_point.path in needed_api_points:
            assert needed_api_points[api_point.path] == api_point.methods
            del needed_api_points[api_point.path]

    assert len(needed_api_points) == 0


@pytest.mark.parametrize(
    'web_app_params, ml_params, error',
    [
        ({'throttling_max_requests': -1}, {},
         'The param throttling_max_requests must be greater than 0. Now it is -1.'),
        ({'throttling_max_requests': 0}, {},
         'The param throttling_max_requests must be greater than 0. Now it is 0.'),
        ({'throttling_max_requests': 0.99}, {},
         'The param throttling_max_requests must be greater than 0. Now it is 0.99.'),
        ({'throttling_max_request_len': -1}, {},
         'The param throttling_max_request_len must be greater than 0. Now it is -1.'),
        ({'throttling_max_request_len': 0}, {},
         'The param throttling_max_request_len must be greater than 0. Now it is 0.'),
        ({'throttling_max_request_len': 0.9}, {},
         'The param throttling_max_request_len must be greater than 0. Now it is 0.9.'),
        ({'column_validation': True}, {'columns': None},
         'The param column_validation=True must use only, when there is ml.columns. Now ml.columns is None.'),
        ({'column_validation': True, 'custom_column_pydantic_model': 123}, {'columns': [{'name': '123'}]},
         'Only one of the two parameters can be used: column_validation, custom_column_pydantic_model. '
         'Now set column_validation=True, custom_column_pydantic_model=123.'),
    ]
)
def test_load_web_app_validation(print_model, web_app_params: Dict, ml_params: Dict, error: str):
    mlup_model = MLupModel(ml_model=print_model, conf=ModelConfig(**ml_params))
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(**web_app_params)
    )
    mlup_model.load()
    try:
        mlup_web_app.load()
        pytest.fail('Not raised error')
    except WebAppLoadError as e:
        assert str(e) == error


def test_load_web_app_with_not_loaded_ml_model(print_model):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(ml=mlup_model)

    try:
        mlup_web_app.load()
        pytest.fail('Not raised error')
    except WebAppLoadError as e:
        assert str(e) == 'Model not loaded to memory. Analyze impossible. Please call ml.load().'


def test_run_web_app_with_not_loaded_ml_model(print_model):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(ml=mlup_model)

    try:
        mlup_web_app.run()
        pytest.fail('Not raised error')
    except WebAppLoadError as e:
        assert str(e) == 'ML Model not loaded to memory. For run web app, please call ml.load(), or ml.load().'


def test_run_web_app_with_not_loaded_web_app(print_model):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(ml=mlup_model)
    mlup_model.load()

    try:
        mlup_web_app.run()
        pytest.fail('Not raised error')
    except WebAppLoadError as e:
        assert str(e) == 'For run web app, please call web.load(), or web.load().'


@pytest.mark.parametrize(
    'debug, expected_json',
    [(False, DEFAULT_API_INFO_METHOD_DATA),  (True, DEFAULT_API_DEBUG_INFO_METHOD_DATA)],
    ids=['debug=False', 'debug=True']
)
@pytest.mark.asyncio
async def test_web_app_api_method_info(web_app_test_client, print_model, debug, expected_json):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(ml=mlup_model, conf=WebAppConfig(debug=debug))
    mlup_model.load()
    mlup_web_app.load()

    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.get("/info")
        assert response.status_code == 200
        assert response.json() == expected_json
        assertDictEqual(response.json(), expected_json)


@pytest.mark.asyncio
async def test_run_with_daemon_is_False(print_model):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(ml=mlup_model, conf=WebAppConfig(host='0.0.0.0', port=8009))
    mlup_model.load()
    mlup_web_app.load()

    mlup_web_app.conf.uvicorn_kwargs['loop'] = 'none'

    web_app_thread = Thread(
        target=mlup_web_app.run,
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
        mlup_web_app._uvicorn_server.should_exit = True
        run_async(
            asyncio.wait_for,
            mlup_web_app._uvicorn_server.shutdown(),
            mlup_web_app.conf.timeout_for_shutdown_daemon
        )
        web_app_thread.join(timeout=3)


@pytest.mark.asyncio
async def test_run_with_daemon_is_True(print_model):
    mlup_model = MLupModel(ml_model=print_model)
    mlup_web_app = MLupWebApp(ml=mlup_model, conf=WebAppConfig(host='0.0.0.0', port=8011))
    mlup_model.load()
    mlup_web_app.load()

    mlup_web_app.run(daemon=True)
    await asyncio.sleep(0.5)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('http://0.0.0.0:8011/health')
            assert response.status_code == 200
            assert response.json() == {'status': 200}
    finally:
        mlup_web_app.stop(shutdown_timeout=3)


@pytest.mark.asyncio
async def test_predict(web_app_test_client, print_model):
    mlup_model = MLupModel(
        ml_model=print_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(mode=WebAppArchitecture.directly_to_predict)
    )
    mlup_model.load()
    mlup_web_app.load()
    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.post("/predict", json={'X': [[1, 2, 3]]})
        assert response.status_code == 200
        assert response.headers['x-predict-id']
        assert response.json() == {"predict_result": [[1, 2, 3]]}


@pytest.mark.asyncio
async def test_http_health(web_app_test_client, print_model):
    mlup_model = MLupModel(
        ml_model=print_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(mode=WebAppArchitecture.directly_to_predict)
    )
    mlup_model.load()
    mlup_web_app.load()
    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": 200}


@pytest.mark.parametrize(
    'show_docs, expected_status',
    [(True, 200), (False, 404)],
    ids=['show_docs=True', 'show_docs=False']
)
@pytest.mark.asyncio
async def test_api_docs(web_app_test_client, print_model, show_docs, expected_status):
    mlup_model = MLupModel(
        ml_model=print_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(mode=WebAppArchitecture.directly_to_predict, show_docs=show_docs)
    )
    mlup_model.load()
    mlup_web_app.load()
    with web_app_test_client(mlup_web_app) as api_test_client:
        response_docs = await api_test_client.get("/docs")
        response_redoc = await api_test_client.get("/redoc")
        assert response_docs.status_code == expected_status
        assert response_redoc.status_code == expected_status


@pytest.mark.asyncio
async def test_predict_model_numpy_returned_valid(web_app_test_client, list_to_numpy_array_model):
    mlup_model = MLupModel(
        ml_model=list_to_numpy_array_model,
        conf=ModelConfig(
            data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            data_transformer_for_predicted=ModelDataTransformerType.NUMPY_ARR,
        )
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(mode=WebAppArchitecture.directly_to_predict)
    )
    mlup_model.load()
    mlup_web_app.load()
    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.post("/predict", json={'X': [[1, 2, 3]]})
        assert response.status_code == 200
        assert response.headers['x-predict-id']
        assert response.json() == {"predict_result": [[1, 2, 3]]}


@pytest.mark.asyncio
async def test_predict_model_numpy_returned_not_valid(web_app_test_client, list_to_numpy_array_model):
    mlup_model = MLupModel(
        ml_model=list_to_numpy_array_model,
        conf=ModelConfig(
            data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            data_transformer_for_predicted=ModelDataTransformerType.PANDAS_DF,
        )
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(mode=WebAppArchitecture.directly_to_predict)
    )
    mlup_model.load()
    mlup_web_app.load()
    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.post("/predict", json={'X': [[1, 2, 3]]})
        assert response.status_code == 500
        assert response.headers['x-predict-id']
        assert response.json() == {
            'detail': [
                {
                    'loc': [],
                    'msg': "'numpy.ndarray' object has no attribute 'to_dict'",
                    'type': 'predict_transform_data_error'
                }
            ],
            'predict_id': response.headers['x-predict-id']
        }


@pytest.mark.asyncio
async def test_predict_not_valid_request(web_app_test_client, print_model):
    mlup_model = MLupModel(
        ml_model=print_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(mode=WebAppArchitecture.directly_to_predict)
    )
    mlup_model.load()
    mlup_web_app.load()
    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.post("/predict", json={'not_exists_key': [[1, 2, 3]], 'test_param': 123})
        assert response.status_code == 422
        assert response.headers['x-predict-id']
        assert response.json() == {
            "detail": [
                {"loc": ["X"], "msg": "field required", "type": "value_error.missing"},
                {"loc": ["test_param"], "msg": "value could not be parsed to a boolean", "type": "type_error.bool"}
            ],
            "predict_id": response.headers['x-predict-id']
        }


@pytest.mark.asyncio
async def test_predict_max_requests_throttling(web_app_test_client, print_sleep_model):
    mlup_model = MLupModel(
        ml_model=print_sleep_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(
            mode=WebAppArchitecture.directly_to_predict,
            throttling_max_requests=1,
        )
    )
    mlup_model.load()
    mlup_model.model_obj.sleep = 0.2
    mlup_web_app.load()
    with web_app_test_client(mlup_web_app) as api_test_client:
        first_request_task = create_async_task(
            api_test_client.post("/predict", json={'X': [[1, 2, 3]]}),
            name='test_predict_max_requests_throttling'
        )
        await asyncio.sleep(0.1)
        try:
            response = await api_test_client.post("/predict", json={'X': [[1, 2, 3]]})
            assert response.status_code == 429
            assert response.headers['x-predict-id']
            assert response.json() == {
                "detail": [
                    {"loc": [], "msg": "Max requests in app. Please try again later.", "type": "throttling_error"}
                ],
                "predict_id": response.headers['x-predict-id']
            }
        finally:
            if not first_request_task.cancelled():
                await first_request_task
                first_request_task.cancel()


@pytest.mark.asyncio
async def test_predict_max_request_len_throttling(web_app_test_client, print_model):
    mlup_model = MLupModel(
        ml_model=print_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(
            mode=WebAppArchitecture.directly_to_predict,
            throttling_max_request_len=1,
        )
    )
    mlup_model.load()
    mlup_web_app.load()

    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.post("/predict", json={'X': [[1, 2, 3], [3, 2, 1], [2, 1, 3]]})
        assert response.status_code == 429
        assert response.headers['x-predict-id']
        assert response.json() == {
            "detail": [
                {
                    "loc": [],
                    "msg": "The query exceeded the limit on the number of rows for the predict. "
                           "Please downsize your request.",
                    "type": "throttling_error"
                }
            ],
            "predict_id": response.headers['x-predict-id']
        }


@pytest.mark.asyncio
async def test_predict_transform_error(web_app_test_client, print_model):
    mlup_model = MLupModel(
        ml_model=print_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.PANDAS_DF),
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(mode=WebAppArchitecture.directly_to_predict),
    )
    mlup_model.load()
    mlup_web_app.load()

    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.post("/predict", json={'X': [[1, 2, 3]]})
        assert response.status_code == 500
        assert response.headers['x-predict-id']
        assert response.json() == {
            "detail": [
                {
                    "loc": [],
                    "msg": "If input data have type list, than need set columns.",
                    "type": "predict_transform_data_error"
                }
            ],
            "predict_id": response.headers['x-predict-id']
        }


@pytest.mark.asyncio
async def test_predict_custom_model_error(web_app_test_client, raise_exception_model):
    mlup_model = MLupModel(
        ml_model=raise_exception_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(mode=WebAppArchitecture.directly_to_predict)
    )
    mlup_model.load()
    mlup_web_app.load()

    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.post("/predict", json={'X': [[1, 2, 3]]})
        assert response.status_code == 500
        assert response.headers['x-predict-id']
        assert response.json() == {
            'detail': [{'loc': [], 'msg': 'Test message', 'type': 'predict_error'}],
            'predict_id': response.headers['x-predict-id']
        }


@pytest.mark.asyncio
async def test_get_predict_result(web_app_test_client, print_sleep_model):
    mlup_model = MLupModel(
        ml_model=print_sleep_model,
        conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
    )
    mlup_web_app = MLupWebApp(
        ml=mlup_model,
        conf=WebAppConfig(
            mode=WebAppArchitecture.worker_and_queue,
            is_long_predict=True,
            ttl_client_wait=0.2,
        )
    )
    mlup_model.load()
    mlup_model.model_obj.sleep = 0.5
    mlup_web_app.load()

    with web_app_test_client(mlup_web_app) as api_test_client:
        response = await api_test_client.post("/predict", json={'X': [[1, 2, 3]]})
        assert response.status_code == 200
        assert response.headers['x-predict-id']
        predict_id = response.json()['predict_result']['predict_id']
        assert predict_id

        # Check, model working yet
        response = await api_test_client.get(f'/get-predict/{predict_id}')
        assert response.status_code == 408
        assert response.json() == {
            'detail': [
                {'loc': [], 'msg': 'Response timed out. Repeat request please.', 'type': 'predict_wait_result_error'}
            ],
            'predict_id': predict_id
        }

        # Mock response architecture
        mlup_web_app._architecture_obj.ttl_client_wait = 30
        mlup_web_app._architecture_obj.results_storage[predict_id] = ([[1, 2, 3]], None)

        # Check get results
        response = await api_test_client.get(f'/get-predict/{predict_id}')
        assert response.status_code == 200
        assert response.json() == [[1, 2, 3]]
