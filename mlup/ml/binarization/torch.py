import logging
from io import BytesIO, BufferedIOBase

import torch

from mlup.errors import ModelBinarizationError
from mlup.ml.binarization.base import BaseBinarizer
from mlup.constants import LoadedFile
from mlup.utils.profiling import TimeProfiler


logger = logging.getLogger('mlup')


class TorchBinarizer(BaseBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info(f'Run deserialization torch data.')
        with TimeProfiler('Time to deserialization torch data:'):
            try:
                _data = data.raw_data
                if data.path:
                    _data = open(data.path, 'rb')
                elif not isinstance(data.raw_data, BufferedIOBase):
                    _data = BytesIO(data.raw_data)
                model = torch.load(_data)
                model.eval()
                return model
            except Exception as e:
                raise ModelBinarizationError(f'Error with deserialize model: {e}')
            finally:
                if hasattr(_data, 'close'):
                    _data.close()

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
                if isinstance(loaded_file.raw_data, str):
                    if loaded_file.raw_data.endswith('.pth'):
                        probability += 0.3
                elif isinstance(loaded_file.raw_data, bytes):
                    if loaded_file.raw_data.endswith(b'.pth'):
                        probability += 0.3
        return probability


class TorchJITBinarizer(BaseBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info(f'Run deserialization torch JIT data.')
        with TimeProfiler('Time to deserialization torch JIT data:'):
            try:
                _data = data.raw_data
                if data.path:
                    _data = open(data.path, 'rb')
                elif not isinstance(data.raw_data, BufferedIOBase):
                    _data = BytesIO(data.raw_data)
                model = torch.jit.load(_data)
                model.eval()
                return model
            except Exception as e:
                raise ModelBinarizationError(f'Error with deserialize model: {e}')
            finally:
                if hasattr(_data, 'close'):
                    _data.close()
