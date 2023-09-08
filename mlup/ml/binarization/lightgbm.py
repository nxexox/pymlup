import logging

import lightgbm as lgb

from mlup.errors import ModelBinarizationError
from mlup.ml.binarization.base import BaseBinarizer
from mlup.constants import LoadedFile
from mlup.utils.profiling import TimeProfiler


logger = logging.getLogger('mlup')


class LightGBMBinarizer(BaseBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info(f'Run deserialization lightgbm data.')
        with TimeProfiler('Time to deserialization lightgbm data:'):
            try:
                if data.path:
                    return lgb.Booster(model_file=data.path)

                if not isinstance(data.raw_data, str):
                    if hasattr(data.raw_data, 'read'):
                        data.raw_data = data.raw_data.read()
                    if isinstance(data.raw_data, bytes):
                        data.raw_data = data.raw_data.decode('utf-8')
                return lgb.Booster(model_str=data.raw_data)
            except Exception as e:
                raise ModelBinarizationError(f'Error with deserialize model: {e}')

    @classmethod
    def _check_type_by_bytes(cls, file_data: str) -> bool:
        # Find version in first rows
        *rows, other = file_data.split('\n', 5)
        return any([r.split('=')[0] == 'version' for r in rows])

    @classmethod
    def is_this_type(cls, loaded_file: LoadedFile) -> float:
        probability = 0.0
        if loaded_file.raw_data:
            file_data = loaded_file.raw_data
            try:
                if isinstance(loaded_file.raw_data, bytes):
                    file_data = loaded_file.raw_data.decode('utf-8')
                if cls._check_type_by_bytes(file_data):
                    probability = 0.8
            except Exception:
                pass
        if loaded_file.path:
            file_data = loaded_file.raw_data
            try:
                with open(loaded_file.path, 'r') as f:
                    file_data = f.read(100)
            except Exception:
                pass
            try:
                if cls._check_type_by_bytes(file_data):
                    probability = 0.8
            except Exception:
                pass
            if str(loaded_file.path).endswith('.txt'):
                probability += 0.05
        return probability
