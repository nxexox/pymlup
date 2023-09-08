import logging
import os.path
import tempfile

import tensorflow

from mlup.errors import ModelBinarizationError
from mlup.ml.binarization.base import BaseBinarizer
from mlup.constants import LoadedFile
from mlup.utils.profiling import TimeProfiler


logger = logging.getLogger('mlup')


class TensorFlowBinarizer(BaseBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info(f'Run deserialization tensorflow data.')
        with TimeProfiler('Time to deserialization tensorflow data:'):
            try:
                _data = data.raw_data
                if data.path:
                    _data = data.path
                elif not isinstance(data.raw_data, str):
                    f = tempfile.NamedTemporaryFile(delete=True, suffix='.keras')
                    if isinstance(data.raw_data, bytes):
                        f.write(data.raw_data)
                    else:
                        f.write(data.raw_data.read())
                    f.seek(0)
                    data.path = f.name
                    _data = f.name
                return tensorflow.keras.models.load_model(str(_data))
            except Exception as e:
                raise ModelBinarizationError(f'Error with deserialize model: {e}')

    @classmethod
    def is_this_type(cls, loaded_file: LoadedFile) -> float:
        probability = 0.0
        if loaded_file.raw_data:
            try:
                if isinstance(loaded_file.raw_data, str):
                    first_bytes = 'PK\x03'
                else:
                    first_bytes = b'PK\x03'
                if loaded_file.raw_data[:3] == first_bytes:
                    probability = 0.5
            except Exception:
                pass
        if loaded_file.path:
            try:
                with open(loaded_file.path, 'rb') as f:
                    file_data = f.read(10)
                    if file_data[:3] == b'PK\x03':
                        probability = 0.5
            except Exception:
                if loaded_file.path.endswith('.keras') or loaded_file.path.endswith('.h5'):
                    probability += 0.3
        return probability


class TensorFlowSavedBinarizer(BaseBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info(f'Run deserialization tensorflow SavedModel data.')
        with TimeProfiler('Time to deserialization tensorflow SavedModel data:'):
            try:
                _data = data.raw_data
                if data.path:
                    _data = data.path
                elif not isinstance(data.raw_data, str):
                    f = tempfile.NamedTemporaryFile(delete=True)
                    if isinstance(data.raw_data, bytes):
                        f.write(data.raw_data)
                    else:
                        f.write(data.raw_data.read())
                    f.seek(0)
                    data.path = f.name
                    _data = f.name
                return tensorflow.saved_model.load(str(_data))
            except Exception as e:
                raise ModelBinarizationError(f'Error with deserialize model: {e}')

    @classmethod
    def is_this_type(cls, loaded_file: LoadedFile) -> float:
        probability = 0.0
        if loaded_file.path:
            try:
                if os.path.isdir(loaded_file.path):
                    if any([f.endswith('.pb') for f in os.listdir(loaded_file.path)]):
                        probability = 0.3
            except Exception:
                pass
        return probability
