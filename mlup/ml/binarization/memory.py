import logging

from mlup.ml.binarization.base import BaseBinarizer
from mlup.constants import LoadedFile

logger = logging.getLogger('mlup')


class MemoryBinarizer(BaseBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info(f'Run deserialization memory data.')
        return data.raw_data
