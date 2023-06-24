import asyncio
import logging
from asyncio import Queue, Task
from typing import Dict, Any

from fastapi import FastAPI

from mlup.interfaces import MLupModelInterface
from mlup.utils.collections import TTLOrderedDict
from mlup.utils.crypto import generate_unique_id
from mlup.web_app.architecture.base import BaseWebAppArchitecture
from mlup.web_app.api_collections import WebAppArchitecture

logger = logging.getLogger('MLup')


class WorkerAndQueueArchitecture(BaseWebAppArchitecture):
    fast_api: FastAPI
    mlup_model: MLupModelInterface
    item_id_col_name: str

    # Worker settings
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
    type: WebAppArchitecture = WebAppArchitecture.directly_to_predict

    def __init__(self, **configs):
        self.fast_api = configs.pop('fast_api')
        self.mlup_model = configs.pop('mlup_model')
        self.item_id_col_name = configs.pop('item_id_col_name')
        self.max_queue_size = configs.pop('max_queue_size')
        self.ttl_predicted_data = configs.pop('ttl_predicted_data')
        self.is_long_predict = configs.pop('is_long_predict')

        if self.ttl_predicted_data <= 0:
            raise ValueError(
                f'The param ttl_predicted_data parameter must be greater than 0. '
                f'Now it is {self.ttl_predicted_data}'
            )

        self._running = False

        self.extra = configs

    async def _get_result_from_storage(self, predict_id: str, sleep_time: float = 0.1):
        while True:
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
            predicted_data = await self.mlup_model.predict(**data_for_predict)
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

    async def run(self):
        logger.info('Run model in worker')
        self.worker_process = asyncio.create_task(self._start_worker())

    async def stop(self):
        logger.info('Stop worker')
        logger.info('Stop checking the batch queue...')
        self._running = False
        self.worker_process.cancel()
        await self.worker_process

    def load(self):
        logger.info(
            f'Load {self.type} web app architecture with params:\n'
            f'    max_queue_size={self.max_queue_size}\n'
            f'    ttl_predicted_data={self.ttl_predicted_data}\n'
            f'    is_long_predict={self.is_long_predict}'
        )
        if not self.results_storage:
            self.results_storage = TTLOrderedDict(self.ttl_predicted_data)
        if not self.queries_queue:
            self.queries_queue = Queue(maxsize=self.max_queue_size)

        if self.is_long_predict:
            self.fast_api.add_api_route("/get-predict/{predict_id}", self.get_predict_result, methods=["GET"])

    async def get_predict_result(self, predict_id: str):
        return await self._get_result_from_storage(predict_id)

    async def predict(self, data_for_predict: Dict) -> Any:
        predict_id = generate_unique_id()
        data_for_queue = {
            self.item_id_col_name: predict_id,
            self.model_data_col_name: data_for_predict
        }

        logger.info(f'New request {predict_id} put to queue.')
        await self.queries_queue.put(data_for_queue)
        logger.debug(f'Request {predict_id} wait predicted')

        if self.is_long_predict:
            return {"predict_id": predict_id}

        return await self._get_result_from_storage(predict_id)
