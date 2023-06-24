from asyncio.locks import Lock as AsyncLock
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from functools import partial
from multiprocessing import Lock
from typing import List, Dict, Callable, Optional, Any, Union

from mlup.config import MLupConfig, LOGGING_CONFIG
from mlup.constants import DEFAULT_X_ARG_NAME, ModelLibraryType, BinarizationType, IS_X
from mlup.interfaces import MLupModelInterface
from mlup.ml_model.binarization import model_load_binary
from mlup.ml_model.data_transformers import BaseDataTransformer, ModelDataType, get_data_transformer_by_type
from mlup.errors import ModelLoadError, ModelPredictError, TransformDataError
from mlup.utils.loop import get_running_loop
from mlup.utils.profiling import TimeProfiler
from mlup.utils.interspection import analyze_method_params
from mlup.web_app.app import MLupWebApp


logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger('MLup')


@dataclass(kw_only=True)
class MLupModel(MLupModelInterface):
    """This is main MLup model class.
    Create object MLup with your ML model, set your settings and run your app.

    Args:
        # Information arguments
        name (str): Model name. Use for information about your model.
            Default is 'MyFirstMLupModel'.

        version (str): Model version. Use for information about your model.
            Default is '1.0.0.0'.

        path_to_binary (str): Path to model binaries files. Use for load model from storage.
            If call load with param path_to_model_binary_files, then MLup remembered path from param.
            Default is empty string.

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

        column_validation (bool): Add column validation after run predict in model.
            If this param set is True, then columns should be set.
            If this param set is False, then column validation will not run after call predict method by model.
            Default is False.

        auto_detect_predict_params (bool): If set is True, MLup will parse predict method your model
            and create from predict method arguments, validators and ApiScheme for inner data.
            If set is False, then MLup wait json with {data_for_predict: [..data for predict..]}.
            Default is True.

        predict_method_name (str): Name for predict method by your model.
            This param use for analyze predict method and run predict method in your model.
            Default is "predict".

        binarization_type (BinarizationType): Type binarization model method. Use for load binary model from storage.
            See mlup.constants.BinarizationType for details.
            Default is BinarizationType.PICKLE.

        # Model work settings
        use_thread_loop (bool): Use concurrent.futures.ThreadPoolExecutor for create workers
            and run in workers pool predict in model.
            This useful for not blocking App CPU bound operations on the predict time.
            Default is True.

        max_thread_loop_workers (Optional[int]): Count thread workers in concurrent.futures.ThreadPoolExecutor.
            Param use if use_thread_loop is set True. If not set this param, then will be calculate,
            how calculate concurrent library - min(32, os.cpu_count() + 4).
            Default is None, that is min(32, os.cpu_count() + 4).

        model_data_type_for_predict (ModelDataType): How data type need for model predict method.
            See mlup.ml_model.data_transformers.ModelDataType for details.
            Default ModelDataType.PANDAS_DF.

        model_data_type_for_predicted (ModelDataType): How data type model returned from predict method.
            See mlup.ml_model.data_transformers.ModelDataType for details.
            Default ModelDataType.NUMPY_ARR.

        data_transformer_for_predict (BaseDataTransformer): Class for transform source data to predict method.
            Auto detected by model_data_type_for_predict param.
            If set custom class, will be use your custom class.
            Your custom transform class well need to inherit mlup.ml_model.data_transformers.BaseDataTransformer.
            See mlup.ml_model.data_transformers for details.
            Default is None and aut detected by model_data_type_for_predict.

        data_transformer_for_predicted (BaseDataTransformer): Class for transform predicted data
            from predict method to App return format.
            Auto detected by model_data_type_for_predicted param.
            If set custom class, will be use your custom class.
            Your custom transform class well need to inherit mlup.ml_model.data_transformers.BaseDataTransformer.
            See mlup.ml_model.data_transformers for details.
            Default is None and aut detected by model_data_type_for_predicted.

    """
    name: str = 'MyFirstMLupModel'
    version: str = '1.0.0.0'
    path_to_binary: Optional[str] = None

    # Model Interface settings
    type: ModelLibraryType = ModelLibraryType.SKLEARN
    columns: Optional[List[Dict[str, str]]] = None
    auto_detect_predict_params: bool = True
    predict_method_name: str = 'predict'
    binarization_type: BinarizationType = BinarizationType.PICKLE

    # Model work settings
    use_thread_loop: bool = True
    max_thread_loop_workers: Optional[int] = None
    model_data_type_for_predict: ModelDataType = ModelDataType.PANDAS_DF
    model_data_type_for_predicted: ModelDataType = ModelDataType.NUMPY_ARR
    data_transformer_for_predict: BaseDataTransformer = None
    data_transformer_for_predicted: BaseDataTransformer = None

    # Inner model data
    config: MLupConfig = field(init=False, default=None)
    x_column_name: str = field(init=False, default=None)
    _ml_model_obj = None
    _lock: Lock = field(init=False, default=None)
    _async_lock: AsyncLock = field(init=False, default=None)
    _pool_workers: ThreadPoolExecutor = field(init=False, default=None)
    _predict_method: Callable = field(init=False, default=None)
    _predict_arguments: List[Dict] = field(init=False, default=None)
    _wep_app: MLupWebApp = field(init=False, default=None)

    def __del__(self):
        """Remove ThreadPoolExecutor workers"""
        if self._pool_workers is not None:
            self._pool_workers.shutdown(wait=False)

    def __post_init__(self):
        """Run custom construct code after call dataclass construct code"""
        if self._async_lock is None:
            self._async_lock = AsyncLock()
        if self._lock is None:
            self._lock = Lock()
        if self.config is None:
            self.config = MLupConfig(mlup_model=self, web_app=self.web_app)

    @property
    def ml_model_obj(self):
        """Source model object."""
        if self._ml_model_obj is None:
            raise ModelLoadError('Model object not found. Please call load().')
        return self._ml_model_obj

    @property
    def web_app(self) -> MLupWebApp:
        if not self._wep_app:
            self._wep_app = MLupWebApp(mlup_model=self)
        return self._wep_app

    def load(
        self,
        path_to_model_binary_files: Optional[str] = None,
        ml_model: Optional[Any] = None,
        force_loading: bool = False,
    ):
        """
        Load binary model.
        May be set path_to_model_binary_files or ml_model and not both params.
        If model already loaded, new load will not run. For force loading, set force_loading=True.

        :param Optional[str] path_to_model_binary_files: Path to binary files in disk for load model from disk.
        :param Optional[Callable] ml_model: Model object in memory, Example in jupyter after learning model.
        :param bool force_loading: Reloading model, if model already loaded.

        """
        if sum((path_to_model_binary_files is not None, ml_model is not None)) != 1:
            if not path_to_model_binary_files:
                path_to_model_binary_files = self.path_to_binary
            elif not self.path_to_binary:
                self.path_to_binary = path_to_model_binary_files
            if sum((path_to_model_binary_files is not None, ml_model is not None)) != 1:
                raise ModelLoadError('You can set only one argument: path_to_model_binary_files, ml_model.')

        if path_to_model_binary_files:
            self._load_binary_model_from_disk(path_to_model_binary_files, force_loading)
        elif ml_model:
            self._load_binary_model_from_obj(ml_model, force_loading)

        self._predict_method = self._get_predict_method()
        self._predict_arguments = analyze_method_params(self._predict_method, self.auto_detect_predict_params)
        self.x_column_name = self._get_x_column_name()
        if self.use_thread_loop:
            self._pool_workers = ThreadPoolExecutor(
                self.max_thread_loop_workers, thread_name_prefix='mlup_predict'
            )
        else:
            self._pool_workers = None

    # Load model interface
    def _get_predict_method(self):
        try:
            return getattr(self.ml_model_obj, self.predict_method_name)
        except AttributeError as error:
            raise ModelLoadError(str(error))

    def _get_x_column_name(self) -> str:
        if self.auto_detect_predict_params:
            for col_conf in self._predict_arguments:
                if col_conf.get(IS_X, False):
                    return col_conf['name']
        return DEFAULT_X_ARG_NAME

    def _load_binary_model_from_disk(self, path_to_model_binary_files: str, force_loading: bool = False):
        if self._ml_model_obj is None:
            self.path_to_binary = path_to_model_binary_files
            self._ml_model_obj = model_load_binary(self.type, path_to_model_binary_files, self.binarization_type)
        elif self._ml_model_obj and force_loading:
            self.path_to_binary = path_to_model_binary_files
            self._ml_model_obj = model_load_binary(self.type, path_to_model_binary_files, self.binarization_type)
        else:
            logger.error(
                'The binary model is already loaded into memory. '
                'To force the loading of the binary model, specify the force_loading=True parameter.'
            )

    def _load_binary_model_from_obj(self, ml_model: Callable, force_loading: bool = False):
        if self._ml_model_obj is None:
            self._ml_model_obj = ml_model
            logging.info(f'Success load MLModel to memory from object.')
        elif self._ml_model_obj and force_loading:
            self._ml_model_obj = ml_model
            logging.info(f'Success load MLModel to memory from object.')
        else:
            logger.error(
                f'The binary model is already loaded into memory. '
                f'To force the loading of the binary model, specify the force_loading=True parameter.'
            )

    # Work model interface
    def _get_data_transformer_for_predict(self) -> BaseDataTransformer:
        if isinstance(self.data_transformer_for_predict, BaseDataTransformer):
            return self.data_transformer_for_predict
        self.data_transformer_for_predict = get_data_transformer_by_type(self.model_data_type_for_predict)
        return self.data_transformer_for_predict

    def _get_data_transformer_for_predicted(self) -> BaseDataTransformer:
        if isinstance(self.data_transformer_for_predicted, BaseDataTransformer):
            return self.data_transformer_for_predicted
        self.data_transformer_for_predicted = get_data_transformer_by_type(self.model_data_type_for_predicted)
        return self.data_transformer_for_predicted

    def _transform_data_for_predict(self, src_data: List[Union[Dict, List]]):
        logger.debug(f'Transform data {len(src_data)} rows to model format.')
        with TimeProfiler('The processing data', log_level='debug'):
            try:
                data_transformer = self._get_data_transformer_for_predict()
                processing_data = data_transformer.transform_to_model_format(src_data, self.columns)
            except Exception as e:
                logger.exception(f'Fail transform data rows {len(src_data)} to model format.')
                raise TransformDataError(str(e))
            finally:
                logger.debug(f'Finish transform data {len(src_data)} rows to model format.')
        return processing_data

    def _transform_predicted_data(self, predicted_data: Any):
        logger.debug('Transform predicted data to response format.')
        with TimeProfiler('The processing data', log_level='debug'):
            try:
                data_transformer = self._get_data_transformer_for_predicted()
                processing_data = data_transformer.transform_to_json_format(predicted_data)
            except Exception as e:
                logger.exception(f'Fail transform predicted data to response format.')
                raise TransformDataError(str(e))
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
        loop = get_running_loop()
        _predict_args = (data_for_predict,) if data_for_predict is not None else tuple()

        with TimeProfiler('The model work '):
            try:
                if self.use_thread_loop:
                    logger.debug(f'Run predict in ThreadPoolExecutor({self.max_thread_loop_workers}).')
                    async with self._async_lock:
                        result = await loop.run_in_executor(
                            self._pool_workers,
                            self._prepare_predict_method(**other_predict_args),
                            *_predict_args
                        )
                else:
                    logger.debug(f'Run predict in sync method.')
                    with self._lock:
                        result = self._prepare_predict_method(**other_predict_args)(*_predict_args)
            except Exception as e:
                error = e

        if error:
            raise error

        return result

    def get_X_from_predict_data(self, predict_data: Dict, remove: bool = True) -> Optional[List[Union[Dict, List]]]:
        if self.x_column_name is None:
            raise ModelLoadError('Model object not loaded. Please call load().')
        if remove:
            return predict_data.pop(self.x_column_name, None)
        return predict_data.get(self.x_column_name, None)

    def sync_predict(self, **predict_data):
        loop = get_running_loop()
        return loop.run_until_complete(self.predict(**predict_data))

    async def predict(self, **predict_data):
        if self._ml_model_obj is None:
            raise ModelLoadError('Model object not loaded. Please call load().')
        try:
            src_X = self.get_X_from_predict_data(predict_data)
            data_for_predict = None
            if src_X is not None:
                data_for_predict = self._transform_data_for_predict(src_X)

            predicted_data = await self._predict(data_for_predict, **predict_data)
            return self._transform_predicted_data(predicted_data)
        except TransformDataError:
            raise
        except Exception as e:
            raise ModelPredictError(e)

    def run_web_app(self, daemon: bool = False):
        """
        Run web app with ML model.

        :param bool daemon: Run web app with daemon mode and unblock terminal. Default is False.

        """
        self.web_app.load()
        self.web_app.run_app(daemon=daemon)

    def stop_web_app(self):
        """Stop web app, if web app was runned in daemon mode."""
        self.web_app.stop_app()
