import logging
import pickle
import pickletools
from io import BufferedIOBase

from mlup.errors import ModelBinarizationError
from mlup.ml.binarization.base import BaseBinarizer
from mlup.constants import LoadedFile
from mlup.utils.profiling import TimeProfiler


logger = logging.getLogger('mlup')


class PickleBinarizer(BaseBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info(f'Run deserialization pickle data.')
        with TimeProfiler('Time to deserialization pickle data:'):
            try:
                if isinstance(data.raw_data, BufferedIOBase):
                    return pickle.load(data.raw_data)
                elif data.raw_data:
                    return pickle.loads(data.raw_data)
                else:
                    with open(data.path, 'rb') as f:
                        return pickle.load(f)
            except Exception as e:
                raise ModelBinarizationError(f'Error with deserialize model: {e}')

    @classmethod
    def _check_type_by_bytes(cls, file_bytes: bytes) -> bool:
        # Check by first and end bytes
        # https://stackoverflow.com/a/73523239
        start_opcode = pickletools.code2op.get(file_bytes[0:1].decode('latin-1'))
        end_opcode = pickletools.code2op.get(file_bytes[-1:].decode('latin-1'))
        return start_opcode.name == 'PROTO' and end_opcode.name == 'STOP'

    @classmethod
    def is_this_type(cls, loaded_file: LoadedFile) -> float:
        probability = 0.0
        if loaded_file.raw_data:
            file_bytes = loaded_file.raw_data
            try:
                if isinstance(file_bytes, str):
                    file_bytes = file_bytes.encode()
                if cls._check_type_by_bytes(file_bytes):
                    probability = 0.9
            except Exception:
                pass

        if loaded_file.path:
            try:
                with open(loaded_file.path, 'rb') as f:
                    # Read only first and last bytes, for check
                    start = f.read(1)
                    f.seek(-2, 2)
                    end = f.read()
                    if cls._check_type_by_bytes(start + end):
                        probability = 0.9
            except Exception:
                pass
            finally:
                if str(loaded_file.path).endswith('.pckl') or str(loaded_file.path).endswith('.pkl'):
                    probability += 0.05
        return probability
