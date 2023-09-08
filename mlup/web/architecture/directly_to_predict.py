import logging
from typing import Dict, Any

from mlup.ml.model import MLupModel
from mlup.web.architecture.base import BaseWebAppArchitecture
from mlup.constants import WebAppArchitecture

logger = logging.getLogger('mlup')


class DirectlyToPredictArchitecture(BaseWebAppArchitecture):
    ml: MLupModel
    type: WebAppArchitecture = WebAppArchitecture.directly_to_predict

    def __init__(self, **configs):
        self.ml = configs.pop('ml')
        self.extra = configs

    def load(self):
        logger.info(f'Load {self.type}')

    async def run(self):
        logger.debug('Run model directly to predict')

    @property
    def is_running(self) -> bool:
        return True

    async def stop(self):
        logger.debug('Model directly stopped')

    async def predict(self, data_for_predict: Dict, predict_id: str) -> Any:
        logger.info(f'New request {predict_id}.')
        logger.debug(f'Request {predict_id} wait predicted')
        result = await self.ml.predict(**data_for_predict)
        logger.info(f'End predict for {predict_id}')
        return result
