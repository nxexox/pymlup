import abc
from typing import Dict, Any

from mlup.ml.model import MLupModel
from mlup.constants import WebAppArchitecture


class BaseWebAppArchitecture(metaclass=abc.ABCMeta):
    ml: MLupModel
    type: WebAppArchitecture

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    async def run(self):
        pass

    @property
    @abc.abstractmethod
    def is_running(self) -> bool:
        pass

    @abc.abstractmethod
    async def stop(self):
        pass

    @abc.abstractmethod
    async def predict(self, data_for_predict: Dict, predict_id: str) -> Any:
        pass
