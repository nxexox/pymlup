import asyncio
from asyncio import InvalidStateError
import pickle

import pytest

from mlup.constants import ModelDataTransformerType, BinarizationType, StorageType, WebAppArchitecture
from mlup.up import UP, Config
from mlup.utils.loop import create_async_task

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


@pytest.mark.skipif(lgb is None, reason='lightgbm library not installed.')
class TestLightGBMModel:
    @pytest.mark.parametrize(
        'binarizer_type, model_fixture_name',
        [
            (BinarizationType.PICKLE, 'lightgbm_binary_cls_model'),
            (BinarizationType.LIGHTGBM, 'lightgbm_binary_cls_model_txt'),
        ]
    )
    @pytest.mark.asyncio
    async def test_load_from_source(self, binarizer_type, model_fixture_name, request):
        model_and_path = request.getfixturevalue(model_fixture_name)
        up = UP(
            conf=Config(
                storage_type=StorageType.disk,
                storage_kwargs={'path_to_files': model_and_path.path, 'files_mask': model_and_path.file_mask},
                data_transformer_for_predict=ModelDataTransformerType.PANDAS_DF,
                data_transformer_for_predicted=ModelDataTransformerType.NUMPY_ARR,
                binarization_type=binarizer_type,
            ),
        )
        up.ml.load()
        assert up.predict(**{model_and_path.x_arg_name: [model_and_path.test_data_raw]}) == \
               [model_and_path.test_model_response_raw]

    @pytest.mark.asyncio
    async def test_predict(self, lightgbm_binary_cls_model):
        up = UP(
            ml_model=lightgbm_binary_cls_model.model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.PANDAS_DF,
                data_transformer_for_predicted=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        up.ml.load()

        pred = up.predict(**{lightgbm_binary_cls_model.x_arg_name: [lightgbm_binary_cls_model.test_data_raw]})
        assert pred == [lightgbm_binary_cls_model.test_model_response_raw]

    @pytest.mark.parametrize(
        'use_thread_loop', [True, False],
        ids=['use_thread_loop=True', 'use_thread_loop=False']
    )
    @pytest.mark.asyncio
    async def test_predict_thread_pool_worker(self, lightgbm_binary_cls_model, use_thread_loop):
        up = UP(
            ml_model=lightgbm_binary_cls_model.model,
            conf=Config(
                use_thread_loop=use_thread_loop,
                data_transformer_for_predict=ModelDataTransformerType.PANDAS_DF,
                data_transformer_for_predicted=ModelDataTransformerType.NUMPY_ARR,
            ),
        )
        up.ml.load()

        pred = up.predict(**{lightgbm_binary_cls_model.x_arg_name: [lightgbm_binary_cls_model.test_data_raw]})
        assert pred == [lightgbm_binary_cls_model.test_model_response_raw]

    @pytest.mark.asyncio
    async def test_web_app_run_and_predict(self, lightgbm_binary_cls_model, web_app_test_client):
        up = UP(
            ml_model=lightgbm_binary_cls_model.model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.PANDAS_DF,
                data_transformer_for_predicted=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        up.ml.load()
        up.web.load()

        with web_app_test_client(up.web) as api_test_client:
            response = await api_test_client.post(
                "/predict",
                json={lightgbm_binary_cls_model.x_arg_name: [lightgbm_binary_cls_model.test_data_raw]},
            )
            assert response.status_code == 200
            assert response.json() == {"predict_result": [lightgbm_binary_cls_model.test_model_response_raw]}

    @pytest.mark.asyncio
    async def test_web_app_with_worker_architecture(self, lightgbm_binary_cls_model, web_app_test_client):
        up = UP(
            ml_model=lightgbm_binary_cls_model.model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.PANDAS_DF,
                data_transformer_for_predicted=ModelDataTransformerType.NUMPY_ARR,
                mode=WebAppArchitecture.worker_and_queue,
            )
        )
        up.ml.load()
        up.web.load()

        with web_app_test_client(up.web) as api_test_client:
            response = await api_test_client.post(
                "/predict",
                json={lightgbm_binary_cls_model.x_arg_name: [lightgbm_binary_cls_model.test_data_raw]},
            )
            assert response.status_code == 200
            assert response.json() == {"predict_result": [lightgbm_binary_cls_model.test_model_response_raw]}

    @pytest.mark.asyncio
    async def test_web_app_with_batching_architecture(self, lightgbm_binary_cls_model, web_app_test_client):
        up = UP(
            ml_model=lightgbm_binary_cls_model.model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.PANDAS_DF,
                data_transformer_for_predicted=ModelDataTransformerType.NUMPY_ARR,
                mode=WebAppArchitecture.batching,
                ttl_predicted_data=1,
                batch_worker_timeout=0.1,
                ttl_client_wait=2,
                is_long_predict=True,
            )
        )
        up.ml.load()
        up.web.load()

        with web_app_test_client(up.web) as api_test_client:
            try:
                await asyncio.sleep(0.05)
                pred_resp_1 = await api_test_client.post(
                    "/predict",
                    json={lightgbm_binary_cls_model.x_arg_name: [lightgbm_binary_cls_model.test_data_raw]},
                )
                await asyncio.sleep(1)
                pred_resp_2 = await api_test_client.post(
                    "/predict",
                    json={lightgbm_binary_cls_model.x_arg_name: [lightgbm_binary_cls_model.test_data_raw]},
                )
                pred_id_2 = pred_resp_2.json()["predict_result"]["predict_id"]
                pred_result_resp_2 = await api_test_client.get("/get-predict/" + pred_id_2)
                assert pred_result_resp_2.json() == [lightgbm_binary_cls_model.test_model_response_raw]

                # Not found results. Long client wait response
                pred_id_1 = pred_resp_1.json()["predict_result"]["predict_id"]
                task = create_async_task(api_test_client.get("/get-predict/" + pred_id_1))
                await asyncio.sleep(0.2)
                if task.cancelled():
                    pytest.fail('Tasks could not have status cancelled.')
                task.cancel()
                task.result()
            except InvalidStateError as e:
                assert str(e) == 'Result is not set.'
            finally:
                await asyncio.sleep(1)

    @pytest.mark.asyncio
    async def test_pickle(self, lightgbm_binary_cls_model):
        up = UP(
            ml_model=lightgbm_binary_cls_model.model,
            conf=Config(
                data_transformer_for_predict=ModelDataTransformerType.PANDAS_DF,
                data_transformer_for_predicted=ModelDataTransformerType.NUMPY_ARR,
            )
        )
        up.ml.load()
        pred_before_pickle = up.predict(
            **{lightgbm_binary_cls_model.x_arg_name: [lightgbm_binary_cls_model.test_data_raw]}
        )

        up_binary_data = pickle.dumps(up)
        up_after_pickle = pickle.loads(up_binary_data)
        pred_after_pickle = up_after_pickle.predict(
            **{lightgbm_binary_cls_model.x_arg_name: [lightgbm_binary_cls_model.test_data_raw]}
        )
        assert pred_after_pickle == pred_before_pickle == [lightgbm_binary_cls_model.test_model_response_raw]
