import asyncio
import logging
import time
from asyncio import Queue, Task
from typing import Dict, Any

from fastapi import FastAPI

from mlup.errors import WebAppLoadError, PredictError, PredictWaitResultError
from mlup.ml.model import MLupModel
from mlup.utils.collections import TTLOrderedDict
from mlup.web.architecture.base import BaseWebAppArchitecture
from mlup.constants import WebAppArchitecture

logger = logging.getLogger('mlup')


class WorkerAndQueueArchitecture(BaseWebAppArchitecture):
    fast_api: FastAPI
    ml: MLupModel
    item_id_col_name: str

    # Worker settings
    # Max queue size for waiting predict.
    max_queue_size: int
    # Max time member predict result data. Need for clean if client not returned for predict result.
    ttl_predicted_data: int
    # Added get-predict api method and add return predict_id to predict response.
    is_long_predict: bool
    # Max time wait results for clients, in single request.
    ttl_client_wait: float

    model_data_col_name: str = 'data_for_predict'
    results_storage: Dict = None
    queries_queue: Queue = None
    worker_process: Task = None
    type: WebAppArchitecture = WebAppArchitecture.worker_and_queue

    def __init__(self, **configs):
        self.fast_api = configs.pop('fast_api')
        self.ml = configs.pop('ml')
        self.item_id_col_name = configs.pop('item_id_col_name')
        self.max_queue_size = configs.pop('max_queue_size')
        self.ttl_predicted_data = configs.pop('ttl_predicted_data')
        self.is_long_predict = configs.pop('is_long_predict')
        self.ttl_client_wait = configs.pop('ttl_client_wait')

        if self.ttl_predicted_data <= 0:
            raise ValueError(
                f'The param ttl_predicted_data parameter must be greater than 0. '
                f'Now it is {self.ttl_predicted_data}'
            )

        self._running = False

        self.extra = configs

    async def _get_result_from_storage(self, predict_id: str, sleep_time: float = 0.1):
        _start = time.monotonic()
        while True:
            if (time.monotonic() - _start) >= self.ttl_client_wait:
                raise PredictWaitResultError(
                    'Response timed out. Repeat request please.',
                    predict_id=predict_id,
                )
            if predict_id in self.results_storage.keys():
                predict_task_result, error = self.results_storage.get(predict_id)
                if error is not None:
                    raise error
                return predict_task_result
            await asyncio.sleep(sleep_time)

    async def _predict(self, predict_id: str, data_for_predict: Dict) -> Any:
        predicted_data = None
        error = None
        try:
            predicted_data = await self.ml.predict(**data_for_predict)
        except Exception as e:
            error = e

        self.results_storage[predict_id] = (predicted_data, error)

    async def _start_worker(self, sleep_time: float = 0.1):
        self._running = True
        logger.info('Start checking the queue...')

        while self._running:
            await asyncio.sleep(sleep_time)
            if self.queries_queue.empty():
                continue

            queue_data = await self.queries_queue.get()
            predict_id, data_for_predict = queue_data[self.item_id_col_name], queue_data[self.model_data_col_name]
            logger.info(f'Run predict for {predict_id}')
            await self._predict(predict_id, data_for_predict)
            logger.info(f'End predict for {predict_id}')
        logger.info('Stop checking the queue...')

    async def run(self):
        if self.is_running:
            raise WebAppLoadError('Worker is already running')
        logger.info('Run model in worker')
        if self.results_storage is None or self.queries_queue is None:
            raise WebAppLoadError('Not called .load() in web.load()')
        self.worker_process = asyncio.create_task(self._start_worker())

    @property
    def is_running(self) -> bool:
        return self._running

    async def stop(self):
        if not self.is_running:
            raise WebAppLoadError('Worker not started')
        self._running = False
        if not self.worker_process.cancelled():
            await self.worker_process
            self.worker_process.cancel()
        logger.info('Worker stopped')

    def load(self):
        logger.debug(
            f'Load {WebAppArchitecture.worker_and_queue.value} with params:\n'
            f'    item_id_col_name={self.item_id_col_name}\n'
            f'    max_queue_size={self.max_queue_size}\n'
            f'    ttl_predicted_data={self.ttl_predicted_data}\n'
            f'    is_long_predict={self.is_long_predict}\n'
            f'    ttl_client_wait={self.ttl_client_wait}'
        )
        self.results_storage = TTLOrderedDict(self.ttl_predicted_data)
        self.queries_queue = Queue(maxsize=self.max_queue_size)

        if self.is_long_predict:
            self.fast_api.add_api_route("/get-predict/{predict_id}", self.get_predict_result, methods=["GET"])

    async def get_predict_result(self, predict_id: str):
        return await self._get_result_from_storage(predict_id)

    async def predict(self, data_for_predict: Dict, predict_id: str) -> Any:
        if not self.is_running:
            raise PredictError('Worker not started. Please call web.run()')

        data_for_queue = {
            self.item_id_col_name: predict_id,
            self.model_data_col_name: data_for_predict
        }

        logger.info(f'New request {predict_id} put to queue.')
        try:
            self.queries_queue.put_nowait(data_for_queue)
        except asyncio.queues.QueueFull:
            raise PredictError('Queue is full. Please try later.')
        logger.debug(f'Request {predict_id} wait predicted')

        if self.is_long_predict:
            return {"predict_id": predict_id}

        return await self._get_result_from_storage(predict_id)
