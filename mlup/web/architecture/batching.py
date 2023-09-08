import asyncio
import logging
import time
from asyncio import Queue, Task
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from mlup.errors import PredictWaitResultError, WebAppLoadError, PredictError
from mlup.ml.model import MLupModel
from mlup.utils.collections import TTLOrderedDict
from mlup.web.architecture.base import BaseWebAppArchitecture
from mlup.constants import WebAppArchitecture


logger = logging.getLogger('mlup')


class BatchingSingleProcessArchitecture(BaseWebAppArchitecture):
    fast_api: FastAPI
    ml: MLupModel
    item_id_col_name: str

    # Min batch len for start predict.
    min_batch_len: int
    # Max time for pending before run batching.
    batch_worker_timeout: float
    # Max queue size for waiting batching.
    max_queue_size: int
    # Max time member predict result data. Need for clean if client not returned for predict result.
    ttl_predicted_data: int
    # Max time wait results for clients, in single request.
    ttl_client_wait: float
    # Added get-predict api method and add return predict_id to predict response.
    is_long_predict: bool

    model_data_col_name: str = 'data_for_predict'
    results_storage: Dict = None
    queries_queue: Queue = None
    worker_process: Task = None
    batch_queue: List = None
    batch_predict_ids: List = None
    type: WebAppArchitecture = WebAppArchitecture.batching

    def __init__(self, **configs):
        self.fast_api = configs.pop('fast_api')
        self.ml = configs.pop('ml')
        self.item_id_col_name = configs.pop('item_id_col_name')
        self.min_batch_len = configs.pop('min_batch_len')
        self.batch_worker_timeout = configs.pop('batch_worker_timeout')
        self.max_queue_size = configs.pop('max_queue_size')
        self.ttl_predicted_data = configs.pop('ttl_predicted_data')
        self.ttl_client_wait = configs.pop('ttl_client_wait')
        self.is_long_predict = configs.pop('is_long_predict')

        if self.batch_worker_timeout <= 0:
            raise ValueError(
                f'The param batch_worker_timeout parameter must be greater than 0. '
                f'Now it is {self.batch_worker_timeout}'
            )
        if self.ttl_predicted_data <= 0:
            raise ValueError(
                f'The param ttl_predicted_data parameter must be greater than 0. '
                f'Now it is {self.ttl_predicted_data}'
            )

        self._running = False
        self.batch_queue = []
        self.batch_predict_ids = []

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

    def _save_predicted_data_from_batch(self, predicted_data: List, error: Optional[Exception]):
        current_predict_id_index = 0
        count_saved = 0
        local_result = []
        for i, item in enumerate(predicted_data):
            local_result.append(item)
            predict_id, predict_id_current_max = self.batch_predict_ids[current_predict_id_index]
            if i == count_saved + predict_id_current_max - 1:
                current_predict_id_index += 1
                self.results_storage[predict_id] = (local_result, error)
                count_saved += len(local_result)
                local_result = []

    async def _predict(self):
        logger.debug(f'Run predict for batch')
        predicted_data = self.batch_queue
        error = None
        try:
            predicted_data = await self.ml.predict(
                **{self.ml.x_column_name: predicted_data}
            )
        except Exception as e:
            error = e

        self._save_predicted_data_from_batch(predicted_data, error)
        self.batch_queue.clear()
        self.batch_predict_ids.clear()
        logger.debug(f'End predict for batch')

    async def _start_worker(self, sleep_time: float = 0.1):
        self._running = True
        logger.info('Start checking the batch queue...')
        while self._running:
            await asyncio.sleep(min(sleep_time, self.batch_worker_timeout))
            if not self.queries_queue.empty():
                _count_queue = worked_time = 0
                started_at = time.monotonic()
                while not self._is_batch_done(worked_time):
                    if self.queries_queue.qsize() > 0:
                        queue_data = await self.queries_queue.get()
                        _count_queue += 1
                        predict_id = queue_data[self.item_id_col_name]
                        data_for_predict = queue_data[self.model_data_col_name]
                        self.batch_queue.extend(data_for_predict)
                        self.batch_predict_ids.append((predict_id, len(data_for_predict)))
                    else:
                        await asyncio.sleep(min(sleep_time, self.batch_worker_timeout))
                    worked_time = time.monotonic() - started_at
                logger.info(f'Batch handling started after {worked_time:.2f} seconds for {_count_queue} requests')
                logger.info(f'Batch length {len(self.batch_queue)}.')
                logger.info(f'Predict ids in batch: {"|".join(next(zip(*self.batch_predict_ids)))}')
                await self._predict()
        logger.info('Stop checking the batch queue...')

    def _is_batch_done(self, worked_time: float) -> bool:
        return len(self.batch_queue) >= self.min_batch_len or worked_time >= self.batch_worker_timeout

    def load(self):
        logger.debug(
            f'Load {WebAppArchitecture.batching.value} with params:\n'
            f'    item_id_col_name={self.item_id_col_name}\n'
            f'    min_batch_len={self.min_batch_len}\n'
            f'    max_queue_size={self.max_queue_size}\n'
            f'    batch_worker_timeout={self.batch_worker_timeout}\n'
            f'    ttl_predicted_data={self.ttl_predicted_data}\n'
            f'    ttl_client_wait={self.ttl_client_wait}\n'
            f'    is_long_predict={self.is_long_predict}'
        )
        self.results_storage = TTLOrderedDict(self.ttl_predicted_data)
        self.queries_queue = Queue(maxsize=self.max_queue_size)

        if self.is_long_predict:
            self.fast_api.add_api_route("/get-predict/{predict_id}", self.get_predict_result, methods=["GET"])

    async def run(self):
        if self.is_running:
            raise WebAppLoadError('Worker is already running')
        logger.info('Run model in batching worker')
        if self.results_storage is None or self.queries_queue is None:
            raise WebAppLoadError('Not called .load() in web.load()')
        self.worker_process = Task(self._start_worker())

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
        logger.info('Batching worker stopped')

    async def get_predict_result(self, predict_id: str):
        return await self._get_result_from_storage(predict_id)

    async def predict(self, data_for_predict: Dict, predict_id: str) -> Any:
        if not self.is_running:
            raise PredictError('Worker not started. Please call web.run()')

        x_for_predict = self.ml.get_X_from_predict_data(data_for_predict)

        data_for_queue = {
            self.item_id_col_name: predict_id,
            self.model_data_col_name: x_for_predict,
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
