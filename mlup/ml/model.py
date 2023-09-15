import asyncio
import gc
import logging
import sys
from asyncio.locks import Lock as AsyncLock
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, InitVar
from enum import Enum
from functools import partial
from multiprocessing import Lock
from typing import List, Dict, Callable, Optional, Any, Union

from mlup.config import set_logging_settings, LOGGING_CONFIG
from mlup.constants import (
    DEFAULT_X_ARG_NAME,
    ModelDataTransformerType,
    ModelLibraryType,
    BinarizationType,
    IS_X,
    StorageType, THERE_IS_ARGS, LoadedFile,
)
from mlup.ml.data_transformers.base import BaseDataTransformer
from mlup.errors import ModelLoadError, PredictError, PredictTransformDataError
from mlup.utils.profiling import TimeProfiler
from mlup.utils.interspection import analyze_method_params, get_class_by_path, auto_search_binarization_type


set_logging_settings(LOGGING_CONFIG)
logger = logging.getLogger('mlup')


@dataclass(kw_only=True)
class ModelConfig:
    """
    MlupModel config class. This class have settings for model.

    # Information arguments
    name (str): Model name. Use for information about your model.
        Default is 'MyFirstMLupModel'.

    version (str): Model version. Use for information about your model.
        Default is '1.0.0.0'.

    # Model interface settings
    type (ModelLibraryType): Model type for choose current code for load and run model.
        See mlup.constants.ModelLibraryType for details.
        Default is ModelLibraryType.SKLEARN.

    columns (Optional[List[Dict[str, str]]]): Columns description for model predict method.
        Format List[Dict], Example [{"name": "col1", "type": "str", "required": True, "default": None}]
        name - Column name;
        type - Column type in string: int, str, float, bool.
        required - bool and optional field. By default is True
        default - Any data and optional.
        If you not set columns, then columns validation should be False.

    predict_method_name (str): Name for predict method by your model.
        This param use for analyze predict method and run predict method in your model.
        Default is "predict".

    # Model load settings
    auto_detect_predict_params (bool): If set is True, UP will parse predict method your model
        and create from predict method arguments, validators and ApiScheme for inner data.
        If set is False, then UP wait json with {data_for_predict: [..data for predict..]}.
        Default is True.

    storage_type (StorageType): Type storage, where should the model be loaded from.
        All storages code there is in mlup.ml.storage package.
        This param use in load() method, by it search storage class and use for load model.
        Params for storage class, can set to storage_kwargs.
        If set StorageType.memory, you can set ml to construct, without load from storage.
        Default: StorageType.memory.

    storage_kwargs (Dict): Naming arguments for storage class from storage_type.
        Default: empty Dict.

    binarization_type (BinarizationType): Type binarization model method.
        Use for load binary model from storage. See mlup.constants.BinarizationType for details.
        If set "auto", run auto search binarization type by model raw binary data and model name in storage.
        If binarization_type="auto" found binarization, than set binarization_type to found binarizer class.
        If binarization_type="auto" not found binarization, than raise ModelLoadError.
        Default is "auto".

    # Model work settings
    use_thread_loop (bool): Use concurrent.futures.ThreadPoolExecutor for create workers
        and run in workers pool predict in model.
        This useful for not blocking App CPU bound operations on the predict time.
        Default is True.

    max_thread_loop_workers (Optional[int]): Count thread workers in concurrent.futures.ThreadPoolExecutor.
        Param use if use_thread_loop is set True. If not set this param, then will be calculate,
        how calculate concurrent library - min(32, os.cpu_count() + 4).
        Default is None, that is min(32, os.cpu_count() + 4).

    data_transformer_for_predict (ModelDataTransformerType): How data type need for model predict method.
        See mlup.constants.ModelDataTransformerType for details.
        Default ModelDataTransformerType.NUMPY_ARR.

    data_transformer_for_predicted (ModelDataTransformerType): How data type model returned from predict method.
        See mlup.constants.ModelDataTransformerType for details.
        Default ModelDataTransformerType.NUMPY_ARR.

    dtype_for_predict (Optional[str]): Dtype for data_transformer_for_predict.
        Each data_transformer uses its own way of searching for a dtype by the specified one.
        But as a rule it is getattr([pandas, numpy, torch, tensorflow], dtype).
        The default is None, and the library itself determines which type.

    """
    name: str = 'MyFirstMLupModel'
    version: str = '1.0.0.0'

    # Model Interface settings
    type: Union[str, ModelLibraryType] = ModelLibraryType.SKLEARN
    columns: Optional[List[Dict[str, str]]] = None
    predict_method_name: str = 'predict'

    # Model load settings
    auto_detect_predict_params: bool = True
    storage_type: Union[str, StorageType] = StorageType.memory
    storage_kwargs: Dict = field(default_factory=dict, repr=False)
    binarization_type: Union[str, BinarizationType] = "auto"

    # Model work settings
    use_thread_loop: bool = True
    max_thread_loop_workers: Optional[int] = None
    data_transformer_for_predict: Union[str, ModelDataTransformerType] = ModelDataTransformerType.NUMPY_ARR
    data_transformer_for_predicted: Union[str, ModelDataTransformerType] = ModelDataTransformerType.NUMPY_ARR
    dtype_for_predict: Optional[str] = None

    def ml_dict(self):
        res = {}
        for n, f in ModelConfig.__dataclass_fields__.items():
            if f.repr is True:
                v = getattr(self, n)
                if isinstance(v, Enum):
                    v = v.value
                res[n] = v
        return res

    def ml_str(self, need_spaces: bool = False):
        res = []
        space = '    ' if need_spaces else ''
        for n, f in ModelConfig.__dataclass_fields__.items():
            if f.repr is True:
                v = getattr(self, n)
                if isinstance(v, Enum):
                    v = v.value
                res.append(space + f'{n}={v}')
        return '\n'.join(res)


@dataclass(kw_only=True, repr=True)
class MLupModel(ModelConfig):
    """This is main UP model class.
    Create object UP with your ML model, set your settings and run your app.

    Args:
        ml (Any): Model object. Use for create UP from memory model, without UP load procedures.
            Recommended use for check work and configure your app. This argument set model_obj attribute.
        conf (ModelConfig): Config for model. If not set this class, config will create automatic with default params.

    """
    ml_model: InitVar[Any] = field(default=None, repr=False)
    conf: ModelConfig = field(default_factory=ModelConfig, repr=False)

    # Inner model data
    x_column_name: str = field(init=False, default=None)
    _model_obj = None
    _lock: Lock = field(init=False, default=None, repr=False)
    _async_lock: AsyncLock = field(init=False, default=None, repr=False)
    _pool_workers: ThreadPoolExecutor = field(init=False, default=None, repr=False)
    _predict_method: Callable = field(init=False, default=None, repr=False)
    _predict_arguments: List[Dict] = field(init=False, default=None, repr=False)
    _data_transformer_for_predict: BaseDataTransformer = field(init=False, default=None, repr=False)
    _data_transformer_for_predicted: BaseDataTransformer = field(init=False, default=None, repr=False)

    @property
    def model_obj(self):
        """Source model object."""
        if not self.loaded:
            raise ModelLoadError('Model object not found. Please call load().')
        return self._model_obj

    @property
    def loaded(self) -> bool:
        return self._model_obj is not None

    def __del__(self):
        """Remove ThreadPoolExecutor workers"""
        if self._pool_workers is not None:
            self._pool_workers.shutdown(wait=False, cancel_futures=False)

    def __getstate__(self):
        """Before pickle object"""
        logger.info(f'Running binarization {self}.')
        attributes = self.__dict__.copy()
        attributes.pop('_lock', None)
        attributes.pop('_async_lock', None)
        attributes.pop('_pool_workers', None)
        attributes.pop('_predict_method', None)
        attributes.pop('_predict_arguments', None)
        attributes.pop('x_column_name', None)
        return attributes

    def __setstate__(self, state):
        """After unpickle object"""
        logger.info(f'Running an {self} load from binary data.')
        self.__dict__ = state
        self._async_lock = AsyncLock()
        self._lock = Lock()
        if self.loaded:
            self.load_model_settings()

    def __post_init__(self, ml_model: Optional[Callable] = None):
        """Running custom construct code after call dataclass construct code"""
        if self._async_lock is None:
            self._async_lock = AsyncLock()
        if self._lock is None:
            self._lock = Lock()

        if self.conf is None:
            self.conf = ModelConfig()

        # Settings model from memory
        if StorageType(self.conf.storage_type) == StorageType.memory:
            if not ml_model:
                raise ModelLoadError(f'If you use {self.conf.storage_type}, need set "ml" argument.')

            self.conf.storage_kwargs['model'] = ml_model

    # Load model interface
    def _get_predict_method(self):
        try:
            return getattr(self.model_obj, self.conf.predict_method_name)
        except AttributeError as error:
            raise ModelLoadError(str(error))

    def _get_x_column_name(self) -> str:
        if self.conf.auto_detect_predict_params:
            for col_conf in self._predict_arguments:
                if col_conf.get(IS_X, False) or col_conf.get(THERE_IS_ARGS, False):
                    return col_conf['name']
        return DEFAULT_X_ARG_NAME

    # Work model interface
    def _get_data_transformer_for_predict(self) -> BaseDataTransformer:
        return get_class_by_path(self.conf.data_transformer_for_predict)(
            dtype_name=self.conf.dtype_for_predict
        )

    def _get_data_transformer_for_predicted(self) -> BaseDataTransformer:
        return get_class_by_path(self.conf.data_transformer_for_predicted)(
            dtype_name=self.conf.dtype_for_predict
        )

    def _transform_data_for_predict(self, src_data: List[Union[Dict, List]]):
        logger.debug(f'Transform data {len(src_data)} rows to model format.')
        with TimeProfiler('The processing data:', log_level='debug'):
            try:
                processing_data = self._data_transformer_for_predict.transform_to_model_format(
                    src_data,
                    self.conf.columns,
                )
            except Exception as e:
                logger.exception(f'Fail transform data rows {len(src_data)} to model format.')
                raise PredictTransformDataError(str(e))
            finally:
                logger.debug(f'Finish transform data {len(src_data)} rows to model format.')
        return processing_data

    def _transform_predicted_data(self, predicted_data: Any):
        logger.debug('Transform predicted data to response format.')
        with TimeProfiler('The processing data:', log_level='debug'):
            try:
                processing_data = self._data_transformer_for_predicted.transform_to_json_format(predicted_data)
            except Exception as e:
                logger.exception(f'Fail transform predicted data to response format.')
                raise PredictTransformDataError(str(e))
            finally:
                logger.debug('Finish transform predicted data to response format.')
        return processing_data

    def _prepare_predict_method(self, **other_predict_args):
        if other_predict_args:
            return partial(self._predict_method, **other_predict_args)
        return self._predict_method

    async def _predict(self, data_for_predict: Optional[Any] = None, **other_predict_args):
        logger.debug('The model processing the input data.')
        error = None
        loop = asyncio.get_running_loop()
        _predict_args = (data_for_predict,) if data_for_predict is not None else tuple()

        with TimeProfiler('The model work:'):
            try:
                if self.conf.use_thread_loop:
                    logger.debug(f'Running predict in ThreadPoolExecutor({self.conf.max_thread_loop_workers}).')
                    async with self._async_lock:
                        result = await loop.run_in_executor(
                            self._pool_workers,
                            self._prepare_predict_method(**other_predict_args),
                            *_predict_args
                        )
                else:
                    logger.debug(f'Running sync predict.')
                    with self._lock:
                        result = self._prepare_predict_method(**other_predict_args)(*_predict_args)
            except Exception as e:
                error = e

        if error:
            raise error

        return result

    def load(self, force_loading: bool = False):
        """
        Load binary model.
        If model already loaded, new load will not run. For force loading, set force_loading=True.

        :param bool force_loading: Reloading model, if model already loaded.

        """
        logger.info('Run load model.')
        if self._model_obj and not force_loading:
            raise ModelLoadError('Model is already loaded. For reload use force_reloading=True.')

        logger.debug(f'Load model with settings:\n{self.conf.ml_str(need_spaces=True)}')
        storage_class = get_class_by_path(self.conf.storage_type)
        logger.debug(f'Create storage {storage_class}.')
        storage = storage_class(**self.conf.storage_kwargs)

        with TimeProfiler('Time to load binary model to memory:'):
            try:
                loaded_files: list[LoadedFile] = storage.load()
            except Exception as e:
                raise ModelLoadError(str(e))

        logger.debug(f'Size raw models loaded data: {sys.getsizeof(loaded_files, default=-1)}')

        if StorageType(self.conf.storage_type) == StorageType.memory:
            self._model_obj = loaded_files[0].raw_data
        else:
            logger.debug(f'Search binarizer: "{self.conf.binarization_type}".')
            if self.conf.binarization_type == 'auto':
                binarization_type = auto_search_binarization_type(loaded_files[0])
                if binarization_type is not None:
                    self.conf.binarization_type = binarization_type
                else:
                    raise ModelLoadError(
                        'Binarizer with auto mode not found. Set binarizer manual or change your models.'
                    )
            binarization_class = get_class_by_path(self.conf.binarization_type)
            self._model_obj = binarization_class.deserialize(loaded_files[0])

        logger.info(f'Size deserializing models: {sys.getsizeof(self._model_obj, default=-1)}')

        del loaded_files
        gc.collect()
        self.load_model_settings()

    def load_model_settings(self):
        """Analyze binary model and make config for use model."""
        logger.info('Run load model settings.')
        if not self.loaded:
            raise ModelLoadError('Model not loaded to memory. Analyze impossible. Please call ml.load().')

        with TimeProfiler('Time to load model settings:'):
            self._data_transformer_for_predict = self._get_data_transformer_for_predict()
            self._data_transformer_for_predicted = self._get_data_transformer_for_predicted()
            self._predict_method = self._get_predict_method()
            self._predict_arguments = analyze_method_params(self._predict_method, self.conf.auto_detect_predict_params)
            self.x_column_name = self._get_x_column_name()
            if self.conf.use_thread_loop:
                self._pool_workers = ThreadPoolExecutor(
                    self.conf.max_thread_loop_workers, thread_name_prefix='mlup_predict'
                )
            else:
                self._pool_workers = None

    def get_X_from_predict_data(self, predict_data: Dict, remove: bool = True) -> Optional[List[Union[Dict, List]]]:
        """Get X param and value from predict data by model settings"""
        if not self.loaded:
            raise ModelLoadError('Model object not loaded. Please call load().')
        if remove:
            return predict_data.pop(self.x_column_name, None)
        return predict_data.get(self.x_column_name, None)

    async def predict(self, **predict_data):
        """
        Call model predict from data with python types.
        Example: up.ml.predict(X=[[1, 2, 3], [4, 5, 6]])

        """
        if not self.loaded:
            raise ModelLoadError('Model object not loaded. Please call load().')
        try:
            src_x = self.get_X_from_predict_data(predict_data)
            data_for_predict = None
            if src_x is not None:
                data_for_predict = self._transform_data_for_predict(src_x)

            predicted_data = await self._predict(data_for_predict, **predict_data)
            return self._transform_predicted_data(predicted_data)
        except PredictTransformDataError:
            raise
        except Exception as e:
            raise PredictError(e)

    async def predict_from(self, **predict_data):
        """
        Call model predict from data with library types.
        Example: up.ml.predict_from(X=numpy.array([[1, 2, 3], [4, 5, 6]]))

        """
        if not self.loaded:
            raise ModelLoadError('Model object not loaded. Please call load().')
        try:
            src_x = self.get_X_from_predict_data(predict_data)
            predicted_data = await self._predict(src_x, **predict_data)
            return self._transform_predicted_data(predicted_data)
        except PredictTransformDataError:
            raise
        except Exception as e:
            raise PredictError(e)

