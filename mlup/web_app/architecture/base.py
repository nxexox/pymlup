import abc
from typing import Dict, Any

from mlup.interfaces import MLupModelInterface
from mlup.web_app.api_collections import WebAppArchitecture


class BaseWebAppArchitecture(metaclass=abc.ABCMeta):
    mlup_model: MLupModelInterface
    type: WebAppArchitecture = WebAppArchitecture.directly_to_predict

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    async def run(self):
        pass

    @abc.abstractmethod
    async def stop(self):
        pass

    @abc.abstractmethod
    async def predict(self, data_for_predict: Dict) -> Any:
        pass
