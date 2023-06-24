import abc
from dataclasses import dataclass
import pickle
import logging
from typing import Dict, Type, Union

from mlup.constants import BinarizationType, ModelLibraryType
from mlup.errors import ModelLoadError
from mlup.ml_model.storage import BaseMLupStorage

try:
    import joblib
except ImportError:
    joblib = None

from mlup.utils.profiling import TimeProfiler


logger = logging.getLogger('MLup')


@dataclass
class BaseMlModelBinarization(metaclass=abc.ABCMeta):
    path_to_model: str
    # storage for future use different cloud storages.
    storage: BaseMLupStorage = None
    binarization: BinarizationType = BinarizationType.PICKLE

    @abc.abstractmethod
    def deserialize_model(self):
        pass


@dataclass
class SingleFileBinarization(BaseMlModelBinarization):

    def deserialize_joblib(self):
        try:
            with open(self.path_to_model, 'rb') as f:
                binary_model = joblib.load(f, 'rb')
        except Exception as e:
            raise ModelLoadError(f'Error with deserialize model: {e}')
        return binary_model

    def deserialize_pickle(self):
        try:
            with open(self.path_to_model, 'rb') as f:
                binary_model = pickle.load(f)
        except Exception as e:
            raise ModelLoadError(f'Error with deserialize model: {e}')
        return binary_model

    def deserialize_model(self):
        logger.info(f'Run load MLModel {self.path_to_model} to memory from disk.')
        try:
            binary_model = None

            if self.binarization == BinarizationType.PICKLE:
                binary_model = self.deserialize_pickle()
            elif self.binarization == BinarizationType.JOBLIB:
                if joblib is None:
                    logger.error('For use joblib, please install it. pip install joblib.')
                    raise ModelLoadError('For use joblib, please install it. pip install joblib.')
                binary_model = self.deserialize_joblib()

            if binary_model is None:
                logger.error(f'Binarization method {self.binarization} not supported.')
                raise ModelLoadError(f'Binarization method {self.binarization} not supported.')

        except Exception:
            logger.error(f'Failed load MLModel {self.path_to_model} to memory from disk.')
            raise
        logger.info(f'Success load MLModel {self.path_to_model} to memory from disk.')
        return binary_model


model_binarization: Dict[ModelLibraryType, Type[BaseMlModelBinarization]] = {
    ModelLibraryType.SKLEARN: SingleFileBinarization,
    ModelLibraryType.SCIKIT_LEARN: SingleFileBinarization,
}


def model_load_binary(
    model_type: Union[str, ModelLibraryType],
    path_to_binaries: str,
    binarization_type: BinarizationType = BinarizationType.PICKLE,
):
    if isinstance(model_type, str):
        try:
            model_type = ModelLibraryType(model_type)
        except ValueError as e:
            raise KeyError(e)

    with TimeProfiler('Time to load binary model:'):
        try:
            binarizer = model_binarization[model_type]
        except KeyError:
            logger.error(f'Model type {model_type} not supported.')
            raise

        return binarizer(
            path_to_binaries,
            binarization=binarization_type,
        ).deserialize_model()
