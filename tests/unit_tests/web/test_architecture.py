import asyncio
from asyncio import InvalidStateError
import random
import time

import pytest
from fastapi import FastAPI

from mlup.constants import ModelDataTransformerType, WebAppArchitecture
from mlup.errors import ModelLoadError, PredictError, WebAppLoadError, PredictWaitResultError
from mlup.ml.model import MLupModel, ModelConfig
from mlup.utils.collections import TTLOrderedDict
from mlup.utils.interspection import get_class_by_path
from mlup.utils.loop import create_async_task
from mlup.web.architecture.directly_to_predict import DirectlyToPredictArchitecture
from mlup.web.architecture.worker_and_queue import WorkerAndQueueArchitecture
from mlup.web.architecture.batching import BatchingSingleProcessArchitecture


test_predict_id = 'test_predict_id'


def shuffle_array(arr):
    return sorted(arr, key=lambda x: random.randint(0, 10))


@pytest.mark.parametrize(
    'archi_type, expected_class', [
        (WebAppArchitecture.directly_to_predict, DirectlyToPredictArchitecture),
        (WebAppArchitecture.worker_and_queue, WorkerAndQueueArchitecture),
        (WebAppArchitecture.batching, BatchingSingleProcessArchitecture),
    ]
)
def test_get_architecture_by_type(archi_type, expected_class):
    assert get_class_by_path(archi_type) == expected_class


def test_get_architecture_by_type_bad_type():
    try:
        get_class_by_path('not exists architecture type')
        pytest.fail('Not raised KeyError with not exists type')
    except ModuleNotFoundError as e:
        assert str(e) == "No module named 'not exists architecture type'"


def test_get_architecture_by_custom_type():
    data = get_class_by_path('mlup.constants.ModelDataTransformerType')
    assert issubclass(data, ModelDataTransformerType)


class TestDirectlyToPredictArchitecture:
    def test_create_with_empty_params(self):
        try:
            DirectlyToPredictArchitecture()
        except KeyError as e:
            assert str(e) == "'ml'"

    def test_create(self, print_model):
        archi = DirectlyToPredictArchitecture(ml=MLupModel(ml_model=print_model), test_value=123)
        assert archi.extra == dict(test_value=123)

    def test_load(self, print_model):
        archi = DirectlyToPredictArchitecture(ml=MLupModel(ml_model=print_model))
        archi.load()

    @pytest.mark.asyncio
    async def test_predict(self, print_model):
        mlup_model = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        mlup_model.load()
        archi = DirectlyToPredictArchitecture(ml=mlup_model)
        archi.load()
        pred_data = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
        assert pred_data == [[1, 2, 3]]

    @pytest.mark.asyncio
    async def test_predict_error(self, raise_exception_model):
        mlup_model = MLupModel(
            ml_model=raise_exception_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        mlup_model.load()
        archi = DirectlyToPredictArchitecture(ml=mlup_model)
        archi.load()
        try:
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pytest.fail('Not raised error')
        except PredictError as e:
            assert str(e) == 'Test message'

    @pytest.mark.asyncio
    async def test_predict_with_not_loaded_model(self, print_model):
        mlup_model = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        archi = DirectlyToPredictArchitecture(ml=mlup_model)
        archi.load()
        try:
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pytest.fail('Not raised error')
        except ModelLoadError as e:
            assert str(e) == 'Model object not loaded. Please call load().'

    @pytest.mark.asyncio
    async def test_predict_with_not_loaded_archi(self, print_model):
        mlup_model = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        mlup_model.load()
        archi = DirectlyToPredictArchitecture(ml=mlup_model)
        archi.load()
        pred_data = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
        assert pred_data == [[1, 2, 3]]


class TestWorkerAndQueueArchitecture:
    _default_params = dict(
        item_id_col_name='col_name',
        max_queue_size=5,
        ttl_predicted_data=13,
        is_long_predict=True,
        ttl_client_wait=30.0,
        # additional values
        additional_value=123,
    )

    def test_create(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params
        )
        assert archi.fast_api == fast_api
        assert archi.ml == ml
        assert archi.item_id_col_name == 'col_name'
        assert archi.max_queue_size == 5
        assert archi.ttl_predicted_data == 13
        assert archi.is_long_predict is True
        assert archi.extra == dict(additional_value=123)

    def test_create_with_empty_params(self):
        try:
            WorkerAndQueueArchitecture()
            pytest.fail('Error not raised')
        except KeyError as e:
            assert str(e) == "'fast_api'"

    @pytest.mark.parametrize(
        'ttl_predicted_data',
        [0, -1]
    )
    def test_create_with_ttl_predicted_data_less_0(self, print_model, ttl_predicted_data):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        params = self._default_params.copy()
        params['ttl_predicted_data'] = ttl_predicted_data
        try:
            WorkerAndQueueArchitecture(
                fast_api=fast_api,
                ml=ml,
                **params
            )
        except ValueError as e:
            assert str(e) == 'The param ttl_predicted_data parameter must be greater than 0. ' \
                             f'Now it is {ttl_predicted_data}'

    def test_load(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params,
        )
        assert archi.results_storage is None
        assert archi.queries_queue is None
        archi.load()
        assert isinstance(archi.results_storage, TTLOrderedDict)
        assert isinstance(archi.queries_queue, asyncio.Queue)

    def test_reload(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params,
        )
        assert archi.results_storage is None
        assert archi.queries_queue is None
        archi.load()
        assert isinstance(archi.results_storage, TTLOrderedDict)
        assert isinstance(archi.queries_queue, asyncio.Queue)
        archi.results_storage['test_key'] = 'test_value'
        assert archi.queries_queue.qsize() == 0
        archi.queries_queue.put_nowait('test_value')
        assert archi.queries_queue.qsize() == 1

        archi.load()
        assert isinstance(archi.results_storage, TTLOrderedDict)
        assert isinstance(archi.queries_queue, asyncio.Queue)
        assert 'test_key' not in archi.results_storage
        assert archi.queries_queue.qsize() == 0

    def test_load_is_long_predict_is_True(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        params = self._default_params.copy()
        params['is_long_predict'] = True
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        assert any(r.path == '/get-predict/{predict_id}' for r in archi.fast_api.routes)

    def test_load_is_long_predict_is_False(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        params = self._default_params.copy()
        params['is_long_predict'] = False
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        assert all(r.path != '/get-predict/{predict_id}' for r in archi.fast_api.routes)

    @pytest.mark.asyncio
    async def test_run_with_not_loaded(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params,
        )
        try:
            await archi.run()
            pytest.fail('Error not raised')
        except WebAppLoadError as e:
            assert str(e) == 'Not called .load() in web.load()'

    @pytest.mark.asyncio
    async def test_run(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        params = self._default_params.copy()
        params['is_long_predict'] = True
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.1)
        try:
            assert archi.is_running
            try:
                await archi.run()
                pytest.fail('Error not raised')
            except WebAppLoadError as e:
                assert str(e) == 'Worker is already running'
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_stop(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        params = self._default_params.copy()
        params['is_long_predict'] = True
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.1)
        assert archi.is_running
        await archi.stop()
        assert archi.is_running is False

    @pytest.mark.asyncio
    async def test_predict(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = False
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            pred_result = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            assert pred_result == [[1, 2, 3]]
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_with_little_max_queue_size(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['max_queue_size'] = 1
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id + '2')
            pytest.fail('Not raised error')
        except PredictError as e:
            assert str(e) == 'Queue is full. Please try later.'
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_with_little_ttl_predicted_data(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['ttl_predicted_data'] = 1
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            pred_id_1 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            await asyncio.sleep(1)
            pred_id_2 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id + '2')
            pred_result_2 = await archi.get_predict_result(pred_id_2["predict_id"])
            assert pred_result_2 == [[1, 2, 3]]

            # Not found results
            task = create_async_task(
                archi.get_predict_result(pred_id_1["predict_id"]),
                name='test_predict_with_little_ttl_predicted_data',
            )
            await asyncio.sleep(0.2)
            if task.cancelled():
                pytest.fail('Tasks could not have status cancelled.')
            task.cancel()
            task.result()
            pytest.fail('Not raised error')
        except InvalidStateError as e:
            assert str(e) == 'Result is not set.'
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_with_little_ttl_client_wait(self, print_sleep_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_sleep_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        ml.model_obj.sleep = 0.3
        params = self._default_params.copy()
        params['is_long_predict'] = False
        params['ttl_client_wait'] = 0.2
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pytest.fail('Not raised error')
        except PredictWaitResultError as e:
            assert str(e) == 'Response timed out. Repeat request please.'
            assert e.predict_id
            await asyncio.sleep(0.2)
            pred_result = await archi.get_predict_result(e.predict_id)
            assert pred_result == [[1, 2, 3]]
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_without_run(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = False
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        try:
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pytest.fail('Error not raised')
        except PredictError as e:
            assert str(e) == 'Worker not started. Please call web.run()'

    @pytest.mark.asyncio
    async def test_get_result_predict(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = True
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            pred_id_1 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pred_id_2 = await archi.predict(data_for_predict={'X': [[3, 2, 1]]}, predict_id=test_predict_id + '2')
            pred_1 = await archi.get_predict_result(pred_id_1['predict_id'])
            pred_2 = await archi.get_predict_result(pred_id_2['predict_id'])
            assert pred_1 == [[1, 2, 3]]
            assert pred_2 == [[3, 2, 1]]
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_get_result_predict_with_little_ttl_client_wait(self, print_sleep_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_sleep_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        ml.model_obj.sleep = 0.3
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['ttl_client_wait'] = 0.2
        archi = WorkerAndQueueArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            pred_id_1 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            try:
                await archi.get_predict_result(pred_id_1['predict_id'])
                pytest.fail('Not raised error')
            except PredictWaitResultError as e:
                assert str(e) == 'Response timed out. Repeat request please.'
                assert pred_id_1['predict_id'] == e.predict_id
        finally:
            await archi.stop()


class TestBatchingSingleProcessArchitecture:
    _default_params = dict(
        item_id_col_name='col_name',
        min_batch_len=10,
        batch_worker_timeout=1.0,
        max_queue_size=5,
        ttl_predicted_data=13,
        is_long_predict=True,
        ttl_client_wait=30.0,
        # additional values
        additional_value=123,
    )

    def test_create(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params
        )
        assert archi.fast_api == fast_api
        assert archi.ml == ml
        assert archi.item_id_col_name == 'col_name'
        assert archi.min_batch_len == 10
        assert archi.batch_worker_timeout == 1.0
        assert archi.max_queue_size == 5
        assert archi.ttl_predicted_data == 13
        assert archi.is_long_predict is True
        assert archi.extra == dict(additional_value=123)

    def test_create_with_empty_params(self):
        try:
            BatchingSingleProcessArchitecture()
            pytest.fail('Error not raised')
        except KeyError as e:
            assert str(e) == "'fast_api'"

    @pytest.mark.parametrize(
        'param_name, param_value',
        [
            ('batch_worker_timeout', -1),
            ('batch_worker_timeout', 0),
            ('ttl_predicted_data', -1),
            ('ttl_predicted_data', 0)
        ]
    )
    def test_create_with_param_less_0(self, print_model, param_name, param_value):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        params = self._default_params.copy()
        params[param_name] = param_value
        try:
            BatchingSingleProcessArchitecture(
                fast_api=fast_api,
                ml=ml,
                **params
            )
        except ValueError as e:
            assert str(e) == f'The param {param_name} parameter must be greater than 0. ' \
                             f'Now it is {param_value}'

    def test_load(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params,
        )
        assert archi.results_storage is None
        assert archi.queries_queue is None
        archi.load()
        assert isinstance(archi.results_storage, TTLOrderedDict)
        assert isinstance(archi.queries_queue, asyncio.Queue)

    def test_reload(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params,
        )
        assert archi.results_storage is None
        assert archi.queries_queue is None
        archi.load()
        assert isinstance(archi.results_storage, TTLOrderedDict)
        assert isinstance(archi.queries_queue, asyncio.Queue)
        archi.results_storage['test_key'] = 'test_value'
        assert archi.queries_queue.qsize() == 0
        archi.queries_queue.put_nowait('test_value')
        assert archi.queries_queue.qsize() == 1

        archi.load()
        assert isinstance(archi.results_storage, TTLOrderedDict)
        assert isinstance(archi.queries_queue, asyncio.Queue)
        assert 'test_key' not in archi.results_storage
        assert archi.queries_queue.qsize() == 0

    def test_load_is_long_predict_is_True(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        params = self._default_params.copy()
        params['is_long_predict'] = True
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        assert any(r.path == '/get-predict/{predict_id}' for r in archi.fast_api.routes)

    def test_load_is_long_predict_is_False(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        params = self._default_params.copy()
        params['is_long_predict'] = False
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        assert all(r.path != '/get-predict/{predict_id}' for r in archi.fast_api.routes)

    @pytest.mark.asyncio
    async def test_run_with_not_loaded(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params,
        )
        try:
            await archi.run()
            pytest.fail('Error not raised')
        except WebAppLoadError as e:
            assert str(e) == 'Not called .load() in web.load()'

    @pytest.mark.asyncio
    async def test_run(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params,
        )
        archi.load()
        await archi.run()
        try:
            await asyncio.sleep(0.1)
            assert archi.is_running
            try:
                await archi.run()
                pytest.fail('Error not raised')
            except WebAppLoadError as e:
                assert str(e) == 'Worker is already running'
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_stop(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(ml_model=print_model)
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **self._default_params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.1)
        assert archi.is_running
        await archi.stop()
        assert archi.is_running is False

    @pytest.mark.asyncio
    async def test_predict(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = False
        params['batch_worker_timeout'] = 0.1
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            pred_result = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            assert pred_result == [[1, 2, 3]]
        finally:
            await archi.stop()

    @pytest.mark.parametrize('min_batch_len', [2, 5, 10, 50])
    @pytest.mark.asyncio
    async def test_predict_with_min_batch_len(self, print_model, min_batch_len):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['min_batch_len'] = min_batch_len
        params['batch_worker_timeout'] = 100
        params['ttl_client_wait'] = 0.5
        params['max_queue_size'] = 100
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)

        try:
            arr = list(range(10))
            arrs = [shuffle_array(arr)]

            pred_id_1 = await archi.predict(data_for_predict={'X': [arrs[0]]}, predict_id=test_predict_id)
            try:
                await archi.get_predict_result(pred_id_1['predict_id'])
                pytest.fail('Not raised error')
            except PredictWaitResultError as e:
                assert str(e) == 'Response timed out. Repeat request please.'
                assert e.predict_id == pred_id_1['predict_id']
            pred_ids = [pred_id_1]
            for i in range(1, min_batch_len):
                arrs.append(shuffle_array(arr))
                _pred_id = await archi.predict(data_for_predict={'X': [arrs[-1]]}, predict_id=test_predict_id + str(i))
                pred_ids.append(_pred_id)
            for i, pred_id in enumerate(pred_ids):
                pred_res = await archi.get_predict_result(pred_id['predict_id'])
                assert pred_res == [arrs[i]]
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_with_batch_worker_timeout(self, print_sleep_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_sleep_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        ml.model_obj.sleep = 0.1
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['ttl_predicted_data'] = 1
        params['min_batch_len'] = 100
        params['batch_worker_timeout'] = 0.001
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.02)
        try:
            start = time.monotonic()
            pred_id_1 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            await asyncio.sleep(0.01)
            pred_id_2 = await archi.predict(data_for_predict={'X': [[3, 2, 1]]}, predict_id=test_predict_id + '1')
            assert len(archi.batch_queue) == 1
            assert archi.queries_queue.qsize() == 1

            pred_result_1 = await archi.get_predict_result(pred_id_1["predict_id"])
            start_two = time.monotonic()
            pred_result_2 = await archi.get_predict_result(pred_id_2["predict_id"])
            end = time.monotonic()
            assert pred_result_1 == [[1, 2, 3]]
            assert pred_result_2 == [[3, 2, 1]]
            assert end - start < 0.4
            assert end - start_two < 0.25
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_with_little_max_queue_size(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['max_queue_size'] = 1
        params['batch_worker_timeout'] = 0.1
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id + '1')
            pytest.fail('Not raised error')
        except PredictError as e:
            assert str(e) == 'Queue is full. Please try later.'
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_with_little_ttl_predicted_data(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['ttl_predicted_data'] = 1
        params['batch_worker_timeout'] = 0.1
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            pred_id_1 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            await asyncio.sleep(1)
            pred_id_2 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id + '1')
            pred_result_2 = await archi.get_predict_result(pred_id_2["predict_id"])
            assert pred_result_2 == [[1, 2, 3]]

            # Not found results
            task = create_async_task(
                archi.get_predict_result(pred_id_1["predict_id"]),
                name='test_predict_with_little_ttl_predicted_data'
            )
            await asyncio.sleep(0.2)
            if task.cancelled():
                pytest.fail('Tasks could not have status cancelled.')
            task.cancel()
            task.result()
            pytest.fail('Not raised error')
        except InvalidStateError as e:
            assert str(e) == 'Result is not set.'
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_with_little_ttl_client_wait(self, print_sleep_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_sleep_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        ml.model_obj.sleep = 0.3
        params = self._default_params.copy()
        params['is_long_predict'] = False
        params['ttl_client_wait'] = 0.3
        params['batch_worker_timeout'] = 0.01
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pytest.fail('Not raised error')
        except PredictWaitResultError as e:
            assert str(e) == 'Response timed out. Repeat request please.'
            assert e.predict_id
            await asyncio.sleep(0.2)
            pred_result = await archi.get_predict_result(e.predict_id)
            assert pred_result == [[1, 2, 3]]
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_without_run(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = False
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        try:
            await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pytest.fail('Error not raised')
        except PredictError as e:
            assert str(e) == 'Worker not started. Please call web.run()'

    @pytest.mark.asyncio
    async def test_get_result_predict(self, print_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['batch_worker_timeout'] = 0.1
        params['ttl_client_wait'] = 1.0
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.05)
        try:
            pred_id_1 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pred_id_2 = await archi.predict(data_for_predict={'X': [[3, 2, 1]]}, predict_id=test_predict_id + '1')
            pred_1 = await archi.get_predict_result(pred_id_1['predict_id'])
            pred_2 = await archi.get_predict_result(pred_id_2['predict_id'])
            assert pred_1 == [[1, 2, 3]]
            assert pred_2 == [[3, 2, 1]]
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_get_result_predict_with_little_ttl_client_wait(self, print_sleep_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_sleep_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        ml.model_obj.sleep = 0.3
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['ttl_client_wait'] = 0.2
        params['batch_worker_timeout'] = 0.1
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.02)
        try:
            pred_id_1 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            try:
                await archi.get_predict_result(pred_id_1['predict_id'])
                pytest.fail('Not raised error')
            except PredictWaitResultError as e:
                assert str(e) == 'Response timed out. Repeat request please.'
                assert pred_id_1['predict_id'] == e.predict_id
        finally:
            await archi.stop()

    @pytest.mark.asyncio
    async def test_predict_with_many_thick_requests(self, print_sleep_model):
        fast_api = FastAPI()
        ml = MLupModel(
            ml_model=print_sleep_model,
            conf=ModelConfig(data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR)
        )
        ml.load()
        ml.model_obj.sleep = 0.1
        params = self._default_params.copy()
        params['is_long_predict'] = True
        params['ttl_predicted_data'] = 10
        params['ttl_client_wait'] = 1
        params['max_queue_size'] = 100
        params['min_batch_len'] = 10
        params['batch_worker_timeout'] = 0.3
        archi = BatchingSingleProcessArchitecture(
            fast_api=fast_api,
            ml=ml,
            **params,
        )
        archi.load()
        await archi.run()
        await asyncio.sleep(0.02)
        try:
            pred_id_1 = await archi.predict(data_for_predict={'X': [[1, 2, 3]]}, predict_id=test_predict_id)
            pred_id_9 = await archi.predict(
                data_for_predict={'X': [[3, 2, 1] for _ in range(9)]},
                predict_id=test_predict_id + '9'
            )
            pred_id_20 = await archi.predict(
                data_for_predict={'X': [[2, 3, 1] for _ in range(20)]},
                predict_id=test_predict_id + '20'
            )

            pred_res_1 = await archi.get_predict_result(pred_id_1['predict_id'])
            pred_res_9 = await archi.get_predict_result(pred_id_9['predict_id'])
            pred_res_20 = await archi.get_predict_result(pred_id_20['predict_id'])
            assert pred_res_20 == [[2, 3, 1] for _ in range(20)]
            assert pred_res_1 == [[1, 2, 3]]
            assert pred_res_9 == [[3, 2, 1] for _ in range(9)]
        finally:
            await archi.stop()
