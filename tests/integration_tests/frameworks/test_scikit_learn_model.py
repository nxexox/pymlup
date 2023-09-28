import asyncio
from asyncio import InvalidStateError
import pickle

import pytest

from mlup.up import UP, Config
from mlup.constants import WebAppArchitecture
from mlup.utils.loop import create_async_task

try:
    import sklearn
except ImportError:
    sklearn = None


@pytest.mark.skipif(sklearn is None, reason='scikit-learn library not installed.')
class TestScikitLearnModel:
    @pytest.mark.asyncio
    async def test_load_from_source(self, scikit_learn_binary_cls_model):
        up = UP(ml_model=scikit_learn_binary_cls_model.model)
        up.ml.load()

        up.predict(**{scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]})

    @pytest.mark.asyncio
    async def test_predict(self, scikit_learn_binary_cls_model):
        up = UP(ml_model=scikit_learn_binary_cls_model.model)
        up.ml.load()

        pred = up.predict(**{scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]})
        assert pred == [scikit_learn_binary_cls_model.test_model_response_raw]

    @pytest.mark.parametrize(
        'use_thread_loop', [True, False],
        ids=['use_thread_loop=True', 'use_thread_loop=False']
    )
    @pytest.mark.asyncio
    async def test_predict_thread_pool_worker(self, scikit_learn_binary_cls_model, use_thread_loop):
        up = UP(
            ml_model=scikit_learn_binary_cls_model.model,
            conf=Config(use_thread_loop=use_thread_loop),
        )
        up.ml.load()

        pred = up.predict(**{scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]})
        assert pred == [scikit_learn_binary_cls_model.test_model_response_raw]

    @pytest.mark.asyncio
    async def test_web_app_run_and_predict(self, scikit_learn_binary_cls_model, web_app_test_client):
        up = UP(ml_model=scikit_learn_binary_cls_model.model)
        up.ml.load()
        up.web.load()

        with web_app_test_client(up.web) as api_test_client:
            response = await api_test_client.post(
                "/predict",
                json={scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]},
            )
            assert response.status_code == 200
            assert response.json() == {"predict_result": [scikit_learn_binary_cls_model.test_model_response_raw]}

    @pytest.mark.asyncio
    async def test_web_app_with_worker_architecture(self, scikit_learn_binary_cls_model, web_app_test_client):
        up = UP(
            ml_model=scikit_learn_binary_cls_model.model,
            conf=Config(
                mode=WebAppArchitecture.worker_and_queue
            )
        )
        up.ml.load()
        up.web.load()

        with web_app_test_client(up.web) as api_test_client:
            response = await api_test_client.post(
                "/predict",
                json={scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]},
            )
            assert response.status_code == 200
            assert response.json() == {"predict_result": [scikit_learn_binary_cls_model.test_model_response_raw]}

    @pytest.mark.asyncio
    async def test_web_app_with_batching_architecture(self, scikit_learn_binary_cls_model, web_app_test_client):
        up = UP(
            ml_model=scikit_learn_binary_cls_model.model,
            conf=Config(
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
                    json={scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]},
                )
                await asyncio.sleep(1)
                pred_resp_2 = await api_test_client.post(
                    "/predict",
                    json={scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]},
                )
                pred_id_2 = pred_resp_2.json()["predict_result"]["predict_id"]
                pred_result_resp_2 = await api_test_client.get("/get-predict/" + pred_id_2)
                assert pred_result_resp_2.json() == [scikit_learn_binary_cls_model.test_model_response_raw]

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
    async def test_pickle(self, scikit_learn_binary_cls_model):
        up = UP(ml_model=scikit_learn_binary_cls_model.model)
        up.ml.load()
        pred_before_pickle = up.predict(
            **{scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]}
        )

        up_binary_data = pickle.dumps(up)
        up_after_pickle = pickle.loads(up_binary_data)
        pred_after_pickle = up_after_pickle.predict(
            **{scikit_learn_binary_cls_model.x_arg_name: [scikit_learn_binary_cls_model.test_data_raw]}
        )
        assert pred_after_pickle == pred_before_pickle
