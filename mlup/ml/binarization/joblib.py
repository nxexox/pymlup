import logging
from io import BytesIO

from mlup.ml.binarization.pickle import PickleBinarizer
from mlup.constants import LoadedFile
from mlup.utils.profiling import TimeProfiler

try:
    import joblib
except ImportError:
    joblib = None

from mlup.errors import ModelBinarizationError


logger = logging.getLogger('mlup')


class JoblibBinarizer(PickleBinarizer):
    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info(f'Run deserialization joblib data.')
        if joblib is None:
            logger.error('For use joblib, please install it. pip install joblib.')
            raise ModelBinarizationError('For use joblib, please install it. pip install joblib.')

        with TimeProfiler('Time to deserialization joblib data:'):
            data_reader = data.raw_data

            try:
                if data.raw_data and isinstance(data.raw_data, bytes):
                    data_reader = BytesIO(data.raw_data)
                elif data.path:
                    with open(data.path, 'rb') as f:
                        return joblib.load(f)
                return joblib.load(data_reader)
            except Exception as e:
                raise ModelBinarizationError(f'Error with deserialize model: {e}')
