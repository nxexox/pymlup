import asyncio
import sys
from asyncio import InvalidStateError
import pickle

import pytest

from mlup.constants import ModelDataTransformerType, StorageType, BinarizationType, WebAppArchitecture
from mlup.up import UP, Config
from mlup.utils.loop import create_async_task

try:
    import tensorflow
except ImportError:
    tensorflow = None


@pytest.mark.skipif(tensorflow is None, reason='tensorflow library not installed.')
class TestTensorFlowModel:
    @pytest.mark.parametrize(
        'binarizer_type, model_fixture_name',
        [
            (BinarizationType.PICKLE, 'tensorflow_binary_cls_model'),
            (BinarizationType.TENSORFLOW, 'tensorflow_binary_cls_model_keras'),
            (BinarizationType.TENSORFLOW, 'tensorflow_binary_cls_model_h5'),
            (BinarizationType.TENSORFLOW_ZIP, 'tensorflow_binary_cls_model_zip'),
        ]
    )
    @pytest.mark.asyncio
    async def test_load_from_source(self, binarizer_type, model_fixture_name, request):
        if sys.version_info.minor == 7 and model_fixture_name == 'tensorflow_binary_cls_model_zip':
            pytest.skip(f'For Python3.7 keras have version less 2.13 and dont have this model format.')
        model_and_path = request.getfixturevalue(model_fixture_name)
        up = UP(
            conf=Config(
                predict_method_name=model_and_path.predict_method_name,
                storage_type=StorageType.disk,
                storage_kwargs={
                    'path_to_files': str(model_and_path.path),
                    'files_mask': model_and_path.file_mask,
                },
                data_transformer_for_predict=ModelDataTransformerType.TENSORFLOW_TENSOR,
                data_transformer_for_predicted=ModelDataTransformerType.TENSORFLOW_TENSOR,
                binarization_type=binarizer_type,
            ),
        )
        up.ml.load()

        assert up.predict(**{model_and_path.x_arg_name: [model_and_path.test_data_raw]}) == \
               [[model_and_path.test_model_response_raw]]

    @pytest.mark.asyncio
    async def test_predict(self, tensorflow_binary_cls_model):
        up = UP(
            ml_model=tensorflow_binary_cls_model.model,
            conf=Config(
                predict_method_name='__call__',
                data_transformer_for_predict=ModelDataTransformerType.TENSORFLOW_TENSOR,
                data_transformer_for_predicted=ModelDataTransformerType.TENSORFLOW_TENSOR,
            )
        )
        up.ml.load()

        pred = up.predict(**{tensorflow_binary_cls_model.x_arg_name: [tensorflow_binary_cls_model.test_data_raw]})
        assert pred == [[tensorflow_binary_cls_model.test_model_response_raw]]

    @pytest.mark.parametrize(
        'use_thread_loop', [True, False],
        ids=['use_thread_loop=True', 'use_thread_loop=False']
    )
    @pytest.mark.asyncio
    async def test_predict_thread_pool_worker(self, tensorflow_binary_cls_model, use_thread_loop):
        up = UP(
            ml_model=tensorflow_binary_cls_model.model,
            conf=Config(
                use_thread_loop=use_thread_loop,
                predict_method_name='__call__',
                data_transformer_for_predict=ModelDataTransformerType.TENSORFLOW_TENSOR,
                data_transformer_for_predicted=ModelDataTransformerType.TENSORFLOW_TENSOR,
            ),
        )
        up.ml.load()

        pred = up.predict(**{tensorflow_binary_cls_model.x_arg_name: [tensorflow_binary_cls_model.test_data_raw]})
        assert pred == [[tensorflow_binary_cls_model.test_model_response_raw]]

    @pytest.mark.asyncio
    async def test_web_app_run_and_predict(self, tensorflow_binary_cls_model, web_app_test_client):
        up = UP(
            ml_model=tensorflow_binary_cls_model.model,
            conf=Config(
                predict_method_name='__call__',
                data_transformer_for_predict=ModelDataTransformerType.TENSORFLOW_TENSOR,
                data_transformer_for_predicted=ModelDataTransformerType.TENSORFLOW_TENSOR,
            )
        )
        up.ml.load()
        up.web.load()

        with web_app_test_client(up.web) as api_test_client:
            response = await api_test_client.post(
                "/predict",
                json={tensorflow_binary_cls_model.x_arg_name: [tensorflow_binary_cls_model.test_data_raw]},
            )
            assert response.status_code == 200
            assert response.json() == {"predict_result": [[tensorflow_binary_cls_model.test_model_response_raw]]}

    @pytest.mark.asyncio
    async def test_web_app_with_worker_architecture(self, tensorflow_binary_cls_model, web_app_test_client):
        up = UP(
            ml_model=tensorflow_binary_cls_model.model,
            conf=Config(
                predict_method_name='__call__',
                data_transformer_for_predict=ModelDataTransformerType.TENSORFLOW_TENSOR,
                data_transformer_for_predicted=ModelDataTransformerType.TENSORFLOW_TENSOR,
                # Web App
                mode=WebAppArchitecture.worker_and_queue
            )
        )
        up.ml.load()
        up.web.load()

        with web_app_test_client(up.web) as api_test_client:
            response = await api_test_client.post(
                "/predict",
                json={tensorflow_binary_cls_model.x_arg_name: [tensorflow_binary_cls_model.test_data_raw]},
            )
            assert response.status_code == 200
            assert response.json() == {"predict_result": [[tensorflow_binary_cls_model.test_model_response_raw]]}

    @pytest.mark.asyncio
    async def test_web_app_with_batching_architecture(self, tensorflow_binary_cls_model, web_app_test_client):
        up = UP(
            ml_model=tensorflow_binary_cls_model.model,
            conf=Config(
                predict_method_name='__call__',
                data_transformer_for_predict=ModelDataTransformerType.TENSORFLOW_TENSOR,
                data_transformer_for_predicted=ModelDataTransformerType.TENSORFLOW_TENSOR,
                # Web App
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
                    json={tensorflow_binary_cls_model.x_arg_name: [tensorflow_binary_cls_model.test_data_raw]},
                )
                await asyncio.sleep(1)
                pred_resp_2 = await api_test_client.post(
                    "/predict",
                    json={tensorflow_binary_cls_model.x_arg_name: [tensorflow_binary_cls_model.test_data_raw]},
                )
                pred_id_2 = pred_resp_2.json()["predict_result"]["predict_id"]
                pred_result_resp_2 = await api_test_client.get("/get-predict/" + pred_id_2)
                assert pred_result_resp_2.json() == [[tensorflow_binary_cls_model.test_model_response_raw]]

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
    async def test_pickle(self, tensorflow_binary_cls_model):
        up = UP(
            ml_model=tensorflow_binary_cls_model.model,
            conf=Config(
                predict_method_name='__call__',
                data_transformer_for_predict=ModelDataTransformerType.TENSORFLOW_TENSOR,
                data_transformer_for_predicted=ModelDataTransformerType.TENSORFLOW_TENSOR,
            )
        )
        up.ml.load()
        pred_before_pickle = up.predict(
            **{tensorflow_binary_cls_model.x_arg_name: [tensorflow_binary_cls_model.test_data_raw]}
        )

        up_binary_data = pickle.dumps(up)
        up_after_pickle = pickle.loads(up_binary_data)
        pred_after_pickle = up_after_pickle.predict(
            **{tensorflow_binary_cls_model.x_arg_name: [tensorflow_binary_cls_model.test_data_raw]}
        )
        assert pred_after_pickle == pred_before_pickle == [[tensorflow_binary_cls_model.test_model_response_raw]]
