import asyncio
import logging
import time
from asyncio import Queue, Task
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from mlup.interfaces import MLupModelInterface
from mlup.utils.collections import TTLOrderedDict
from mlup.utils.crypto import generate_unique_id
from mlup.web_app.architecture.base import BaseWebAppArchitecture
from mlup.web_app.api_collections import WebAppArchitecture

# TODO: Если включен батчинг, тогда другие параметры переданные в запросе игнорируются.
logger = logging.getLogger('MLup')


class BatchingSingleProcessArchitecture(BaseWebAppArchitecture):
    fast_api: FastAPI
    mlup_model: MLupModelInterface
    item_id_col_name: str

    # Min batch len for start predict.
    min_batch_len: int
    # Max time for pending before run batching.
    batch_worker_timeout: float
    # Max queue size for waiting batching.
    max_queue_size: int
    # Max time member predict result data. Need for clean if client not returned for predict result.
    ttl_predicted_data: float
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
        self.mlup_model = configs.pop('mlup_model')
        self.item_id_col_name = configs.pop('item_id_col_name')
        self.min_batch_len = configs.pop('min_batch_len')
        self.batch_worker_timeout = configs.pop('batch_worker_timeout')
        self.max_queue_size = configs.pop('max_queue_size')
        self.ttl_predicted_data = configs.pop('ttl_predicted_data')
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
        while True:
            if predict_id in self.results_storage.keys():
                predict_task_result, error = self.results_storage.get(predict_id)
                if error is not None:
                    raise error
                return predict_task_result
            await asyncio.sleep(sleep_time)

    def _save_predicted_data_from_batch(self, predicted_data: List, error: Optional[Exception]):
        current_predict_id_index = 0

        local_result = []
        for i, item in enumerate(predicted_data):
            local_result.append(item)
            predict_id, predict_id_current_max = self.batch_predict_ids[current_predict_id_index]
            if i == predict_id_current_max - 1:
                current_predict_id_index += 1
                self.results_storage[predict_id] = (local_result, error)
                local_result = []

    async def _predict(self):
        logger.info(f'Run predict for batch')
        predicted_data = self.batch_queue
        error = None
        try:
            predicted_data = await self.mlup_model.predict(
                **{self.mlup_model.x_column_name: predicted_data}
            )
        except Exception as e:
            error = e

        self._save_predicted_data_from_batch(predicted_data, error)
        self.batch_queue.clear()
        self.batch_predict_ids.clear()
        logger.info(f'End predict for batch')

    async def _start_worker(self, sleep_time: float = 0.1):
        self._running = True
        logger.info('Start checking the batch queue...')
        while self._running:
            await asyncio.sleep(sleep_time)
            if not self.queries_queue.empty():
                count_queue = worked_time = 0
                started_at = time.monotonic()
                while not self._is_batch_done(worked_time):
                    if self.queries_queue.qsize() > 0:
                        queue_data = await self.queries_queue.get()
                        count_queue += 1
                        predict_id = queue_data[self.item_id_col_name]
                        data_for_predict = queue_data[self.model_data_col_name]
                        self.batch_queue.extend(data_for_predict)
                        self.batch_predict_ids.append((predict_id, len(data_for_predict)))
                    else:
                        await asyncio.sleep(sleep_time)
                    worked_time = time.monotonic() - started_at
                logger.info(f'Batch handling started after {worked_time:.2f} seconds for {count_queue} requests')
                logger.info(f'Batch length {len(self.batch_queue)}')
                await self._predict()

    def _is_batch_done(self, worked_time: float) -> bool:
        if len(self.batch_queue) > self.min_batch_len or worked_time >= self.batch_worker_timeout:
            return True
        return False

    def load(self):
        logger.info(
            f'Load {self.type} web app architecture with params:\n'
            f'    min_batch_len={self.min_batch_len}\n'
            f'    max_queue_size={self.max_queue_size}\n'
            f'    batch_worker_timeout={self.batch_worker_timeout}\n'
            f'    ttl_predicted_data={self.ttl_predicted_data}\n'
            f'    is_long_predict={self.is_long_predict}\n'
        )
        if not self.results_storage:
            self.results_storage = TTLOrderedDict(self.ttl_predicted_data)
        if not self.queries_queue:
            self.queries_queue = Queue(maxsize=self.max_queue_size)

        if self.is_long_predict:
            self.fast_api.add_api_route("/get-predict/{predict_id}", self.get_predict_result, methods=["GET"])

    async def run(self):
        logger.info('Run model in batching worker')
        self.worker_process = asyncio.create_task(self._start_worker())

    async def stop(self):
        logger.info('Stop batching worker')
        logger.info('Stop checking the batch queue...')
        self._running = False
        self.worker_process.cancel()
        await self.worker_process

    async def get_predict_result(self, predict_id: str):
        return await self._get_result_from_storage(predict_id)

    async def predict(self, data_for_predict: Dict) -> Any:
        x_for_predict = self.mlup_model.get_X_from_predict_data(data_for_predict)
        predict_id = generate_unique_id()

        data_for_queue = {
            self.item_id_col_name: predict_id,
            self.model_data_col_name: x_for_predict,
        }

        logger.info(f'New request {predict_id} put to queue.')
        await self.queries_queue.put(data_for_queue)
        logger.debug(f'Request {predict_id} wait predicted')

        if self.is_long_predict:
            return {"predict_id": predict_id}

        return await self._get_result_from_storage(predict_id)
