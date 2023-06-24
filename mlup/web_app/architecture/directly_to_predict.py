import logging
from typing import Dict, Any

from mlup.interfaces import MLupModelInterface
from mlup.utils.crypto import generate_unique_id
from mlup.web_app.architecture.base import BaseWebAppArchitecture
from mlup.web_app.api_collections import WebAppArchitecture

logger = logging.getLogger('MLup')


class DirectlyToPredictArchitecture(BaseWebAppArchitecture):
    mlup_model: MLupModelInterface
    type: WebAppArchitecture = WebAppArchitecture.directly_to_predict

    def __init__(self, **configs):
        self.mlup_model = configs.pop('mlup_model')
        self.extra = configs

    def load(self):
        logger.info(f'Load {self.type} web app architecture')

    async def run(self):
        pass

    async def stop(self):
        pass

    async def predict(self, data_for_predict: Dict) -> Any:
        predict_id = generate_unique_id()
        logger.info(f'New request {predict_id} put to queue.')
        logger.debug(f'Request {predict_id} wait predicted')
        result = await self.mlup_model.predict(**data_for_predict)
        logger.info(f'End predict for {predict_id}')
        return result
