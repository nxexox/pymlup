import asyncio
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, wraps
from typing import Type, Optional, Dict

from fastapi import FastAPI, Request as FastAPIRequest, Response as FastAPIResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel as PydanticBaseModel, ValidationError
import uvicorn

from mlup.config import LOGGING_CONFIG
from mlup.constants import ITEM_ID_COL_NAME, WebAppArchitecture, PREDICT_ID_HEADER
from mlup.errors import PredictError, WebAppLoadError, PredictWaitResultError, PredictTransformDataError, \
    PredictValidationInnerDataError
from mlup.ml.model import MLupModel
from mlup.utils.crypto import generate_unique_id
from mlup.utils.interspection import get_class_by_path
from mlup.utils.logging import configure_logging_formatter
from mlup.utils.loop import run_async
from mlup.web.api_docs import openapi_schema
from mlup.web.api_errors import (
    ApiErrorResponse,
    api_exception_handler, ApiRequestError, predict_errors_handler, api_request_error_handler,
)
from mlup.web.api_validators import create_pydantic_predict_model, MlModelPredictRequest
from mlup.web.architecture.base import BaseWebAppArchitecture


logger = logging.getLogger('mlup')


def _requests_throttling(func):
    @wraps(func)
    async def wrap(self, *args, **kwargs):
        if not self.conf.throttling_max_requests:
            return await func(self, *args, **kwargs)

        if self._throttling_max_requests_current_count < self.conf.throttling_max_requests:
            self._throttling_max_requests_current_count += 1
            try:
                return await func(self, *args, **kwargs)
            finally:
                self._throttling_max_requests_current_count -= 1
        else:
            raise ApiRequestError(
                'Max requests in app. Please try again later.',
                status_code=429,
                type='throttling_error',
            )

    return wrap


def _set_predict_id_to_response_headers(func):
    @wraps(func)
    async def wrap(*args, response: FastAPIResponse, **kwargs):
        predict_id = generate_unique_id()
        response.headers[PREDICT_ID_HEADER] = predict_id
        try:
            return await func(*args, response=response, **kwargs)
        except Exception as e:
            e.predict_id = predict_id
            raise

    return wrap


@dataclass(kw_only=True)
class WebAppConfig:
    """
    WebAppConfig config class. This class have settings for web app.

    # WebApp web outer interface settings
    host (str): Web host for web app.
        Default is '0.0.0.0'.

    port (int): Web port for web app.
        Default is 8009.

    web_app_version (str): User version for web_app.
        Using the mlup version may not be nice to users your web app.
        However, you can change the library version without changing the model.
        And then the code of the web application backend actually changes.
        You can use this field to notify users of these changes.
        Default is '1.0.0.0'.

    column_validation (bool): Do I need to create a validator for columns from columns config
        that are sent by the user through the web application?
        Default is False.

    custom_column_pydantic_model (Optional[Type[PydanticBaseModel]]): Custom column pydantic model.
        Use for validation. If set this field, flag column_validation not use. Validation will be always.
        Default is None.

    # WebApp architecture settings
    mode (WebAppArchitecture): Work mode web app. We have three modes for work:
        directly_to_predict, worker_and_queue, batching.
        See mlup.constants.WebAppArchitecture for details.
        Default is WebAppArchitecture.directly_to_predict.

    max_queue_size (int): Max queue size with inner data for model.
        It's use in modes worker_and_queue and batching, for controlling queue inner requests size.
        Default is 100.

    ttl_predicted_data (int): Max time member predict result data.
        It's need for clean memory, if client not returned for predict result.
        Default is 60.

    ttl_client_wait (float): Max time wait results for clients, in single request.
        If the client has not waited for a response within this time, the server responds with a 408 response code
        and the user must make a second request.
        This is necessary so as not to make customers wait forever and all the ensuing consequences.
        Default is 30.0

    min_batch_len (int): Min batch len for start predict.
        Batching is regulated by two parameters: the batch accumulation time and the size of the accumulated batch.
        If one of the parameters has reached its limit, then the batch predictor is launched.
        It's min batch size for call batch predict.
        Default is 10.

    batch_worker_timeout (float): Max time for pending before run batching.
        Batching is regulated by two parameters: the batch accumulation time and the size of the accumulated batch.
        If one of the parameters has reached its limit, then the batch predictor is launched.
        It's max time waiting batch for call batch predict.
        Default is 1.0.

    is_long_predict (bool): Added get-predict api method and add return predict_id to predict response.
        Use only in worker_and_queue and batching modes.
        If set this param, after request to /predict, client get predict_id in response.
        If not set this param, than after ttl_client_wait time, client can't get predict result.
        Default is False.

    # WebApp work settings
    show_docs (bool): Enable API docs in web app by URL /docs.
        Default is True

    debug (bool): Debug mode. Use for FastAPI and for show model configs.
        Default is False

    throttling_max_requests (Optional[int]): Max count simultaneous requests to web app.
        If set None, throttling by requests is disable.
        Default is None

    throttling_max_request_len (Optional[int]): Max count objects to predict in single request.
        If set None, throttling by max items in request is disable.
        Default is None

    timeout_for_shutdown_daemon (float): Wait time for graceful shutdown web app.
        Default is 3.0

    uvicorn_kwargs (Dict): Uvicorn server kwargs arguments.
        Default is {}

    item_id_col_name (str): Column name for unique item_id.
        Use in batching and worker. It need for marks requests in worker by request id.
        Added this key for predicted data, but not thrown into model predict.
        Default is mlup.constants.ITEM_ID_COL_NAME - 'mlup_item_id'

    """
    # WebApp web outer interface settings
    host: str = '0.0.0.0'
    port: int = 8009
    web_app_version: str = '1.0.0.0'

    # Can use only single from two params
    column_validation: bool = False
    custom_column_pydantic_model: Optional[Type[PydanticBaseModel]] = None

    # WebApp architecture settings
    # WebApp work mode settings
    mode: WebAppArchitecture = WebAppArchitecture.directly_to_predict
    # Max queue size for waiting batching.
    max_queue_size: int = 100
    # Max time member predict result data. Need for clean if client not returned for predict result.
    ttl_predicted_data: int = 60
    # Max time wait results for clients, in single request.
    ttl_client_wait: float = 30.0
    # Min batch len for start predict.
    min_batch_len: int = 10
    # Max time for pending before run batching.
    batch_worker_timeout: float = 1.0
    # Added get-predict api method and add return predict_id to predict response.
    is_long_predict: bool = False

    # WebApp work settings
    # Enable API docs in web app
    show_docs: bool = True
    # Debug mode. Use for FastAPI and for show model configs
    debug: bool = False
    # Max count simultaneous requests to web app
    throttling_max_requests: Optional[int] = None
    # Max count objects to predict in single request
    throttling_max_request_len: Optional[int] = None
    # Wait time for graceful shutdown web app
    timeout_for_shutdown_daemon: float = 3.0
    # Uvicorn settings
    uvicorn_kwargs: Dict = field(default_factory=dict, repr=False)

    # Column name for unique item_id
    item_id_col_name: str = ITEM_ID_COL_NAME

    def wb_dict(self):
        res = {}
        for n, f in WebAppConfig.__dataclass_fields__.items():
            if f.repr is True:
                v = getattr(self, n)
                if isinstance(v, Enum):
                    v = v.value
                res[n] = v
        return res

    def wb_str(self, need_spaces: bool = False):
        res = []
        space = '    ' if need_spaces else ''
        for n, f in WebAppConfig.__dataclass_fields__.items():
            if f.repr is True:
                v = getattr(self, n)
                if isinstance(v, Enum):
                    v = v.value
                res.append(space + f'{n}={v}')
        return '\n'.join(res)


@dataclass(kw_only=True, repr=True)
class MLupWebApp:
    """This is main UP web app class.
    Create object UP with your ML model, set your settings and run your web app.

    Args:
        ml (MLupModel): mlup model object.
        conf (WebAppConfig): Config for web app.
            If not set this class, config will create automatic with default params.

    """
    ml: MLupModel = field(repr=False)
    conf: WebAppConfig = field(default_factory=WebAppConfig, repr=False)

    # WebApp inner settings
    _fast_api: FastAPI = field(init=False, default=None, repr=False)
    _throttling_max_requests_current_count: int = field(init=False, default=0, repr=False)
    _predict_inner_pydantic_model: Type[MlModelPredictRequest] = field(init=False, repr=False)
    _daemon_thread: threading.Thread = field(init=False, default=None, repr=False)
    _uvicorn_server: uvicorn.Server = field(init=False, repr=False)
    _architecture_obj: BaseWebAppArchitecture = field(init=False, repr=False)

    def __del__(self):
        self.stop()

    def __getstate__(self):
        """Before pickle object"""
        logger.info(f'Running binarization {self}.')
        attributes = self.__dict__.copy()
        attributes.pop('_fast_api', None)
        attributes.pop('_predict_inner_pydantic_model', None)
        attributes.pop('_daemon_thread', None)
        attributes.pop('_uvicorn_server', None)
        attributes.pop('_architecture_obj', None)
        return attributes

    def __setstate__(self, state):
        """After unpickle object"""
        logger.info(f'Running an {self} load from binary data.')
        self.__dict__ = state
        self.load()

    @property
    def loaded(self) -> bool:
        return self._fast_api is not None

    @property
    def app(self) -> FastAPI:
        if not self.loaded:
            raise WebAppLoadError('web nor creating. Please call web.load().')
        return self._fast_api

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Startup web app code
        await self._architecture_obj.run()
        yield
        # Shutdown web app code
        await self._architecture_obj.stop()
        configure_logging_formatter('default')

    def _request_len_throttling(self, data_for_predict: Dict):
        x_for_predict = self.ml.get_X_from_predict_data(data_for_predict, remove=False)
        if x_for_predict and len(x_for_predict):
            raise ApiRequestError(
                'The query exceeded the limit on the number of rows for the predict. Please downsize your request.',
                status_code=429,
                type='throttling_error'
            )

    def _create_app(self):
        fast_api_kwargs = {}
        if self.conf.show_docs is False:
            fast_api_kwargs['docs_url'] = None
            fast_api_kwargs['redoc_url'] = None
        self._fast_api = FastAPI(
            debug=self.conf.debug,
            title=f"MLup web application with model: {self.ml.conf.name} v{self.ml.conf.version}.",
            description=f"Web application for use {self.ml.conf.name} v{self.ml.conf.version} in web.",
            version=self.conf.web_app_version,
            lifespan=self._lifespan,
            exception_handlers={
                RequestValidationError: api_exception_handler,
                ValidationError: api_exception_handler,
                ApiRequestError: api_request_error_handler,

                # Predict errors
                PredictError: predict_errors_handler,
                PredictWaitResultError: predict_errors_handler,
                PredictTransformDataError: predict_errors_handler,
                PredictValidationInnerDataError: predict_errors_handler,
            },
            responses={
                422: {
                    "description": "Error with validation input data",
                    "model": ApiErrorResponse,
                },
                429: {
                    "description": "Throttling input request error",
                    "model": ApiErrorResponse,
                },
                500: {
                    "description": "Error with predict process exception",
                    "model": ApiErrorResponse,
                }
            },
            **fast_api_kwargs,
        )

        # Set web api points
        self.app.add_api_route("/health", self.http_health, methods=["HEAD", "GET", "OPTIONS"])
        if self.conf.debug:
            self.app.add_api_route("/info", self.debug_info, methods=["GET"], name='info')
        else:
            self.app.add_api_route("/info", self.info, methods=["GET"], name='info')
        self.app.add_api_route("/predict", self.predict, methods=["POST"])

        self._architecture_obj = get_class_by_path(self.conf.mode)(
            fast_api=self._fast_api,
            ml=self.ml,
            item_id_col_name=self.conf.item_id_col_name,
            # Worker settings
            max_queue_size=self.conf.max_queue_size,
            ttl_predicted_data=self.conf.ttl_predicted_data,
            # Batching settings
            min_batch_len=self.conf.min_batch_len,
            batch_worker_timeout=self.conf.batch_worker_timeout,
            # Wait result from worker settings
            is_long_predict=self.conf.is_long_predict,
            ttl_client_wait=self.conf.ttl_client_wait,
        )
        self._predict_inner_pydantic_model = create_pydantic_predict_model(
            self.ml,
            self.conf.column_validation,
            custom_column_pydantic_model=self.conf.custom_column_pydantic_model,
        )

    def _run(self, in_thread: bool = False):
        if 'loop' not in self.conf.uvicorn_kwargs:
            self.conf.uvicorn_kwargs['loop'] = 'none' if in_thread else 'auto'
        if 'timeout_graceful_shutdown' not in self.conf.uvicorn_kwargs:
            self.conf.uvicorn_kwargs['timeout_graceful_shutdown'] = self.conf.timeout_for_shutdown_daemon
        if 'log_config' not in self.conf.uvicorn_kwargs:
            self.conf.uvicorn_kwargs['log_config'] = LOGGING_CONFIG
        configure_logging_formatter('web')

        self._uvicorn_server = uvicorn.Server(
            uvicorn.Config(self.app, **self.conf.uvicorn_kwargs),
        )
        self._uvicorn_server.run()

    def load_web_app_settings(self):
        """Load web app settings"""
        self.conf.uvicorn_kwargs['host'] = self.conf.host
        self.conf.uvicorn_kwargs['port'] = self.conf.port
        self.conf.uvicorn_kwargs['timeout_graceful_shutdown'] = int(self.conf.timeout_for_shutdown_daemon)

    def load(self):
        """Create and full load web app"""
        logger.info('Run load Web application')
        if not self.ml.loaded:
            raise WebAppLoadError('Model not loaded to memory. Analyze impossible. Please call ml.load().')

        logger.debug(f'Load Web application with settings:\n{self.conf.wb_str(need_spaces=True)}')
        if self.conf.throttling_max_requests is not None and self.conf.throttling_max_requests < 1:
            raise WebAppLoadError(
                'The param throttling_max_requests must be greater than 0. '
                f'Now it is {self.conf.throttling_max_requests}.'
            )
        if self.conf.throttling_max_request_len is not None and self.conf.throttling_max_request_len < 1:
            raise WebAppLoadError(
                'The param throttling_max_request_len must be greater than 0. '
                f'Now it is {self.conf.throttling_max_request_len}.'
            )

        if self.conf.column_validation and not self.ml.conf.columns:
            raise WebAppLoadError(
                'The param column_validation=True must use only, when there is ml.columns. '
                f'Now ml.columns is {self.ml.conf.columns}.'
            )
        if self.conf.column_validation and self.conf.custom_column_pydantic_model:
            raise WebAppLoadError(
                'Only one of the two parameters can be used: column_validation, custom_column_pydantic_model. '
                f'Now set column_validation={self.conf.column_validation}, '
                f'custom_column_pydantic_model={self.conf.custom_column_pydantic_model}.'
            )

        self.load_web_app_settings()
        self._create_app()
        self._architecture_obj.load()
        self.app.openapi = partial(openapi_schema, app=self.app, ml=self.ml)

    def stop(self, shutdown_timeout: Optional[float] = None):
        """
        Stop web app, if web app was runned in daemon mode.

        :param Optional[float] shutdown_timeout: Wait web_app gracefull shutdown.
            Use this param or self.conf.timeout_for_shutdown_daemon.

        """
        if self._daemon_thread and self._daemon_thread.is_alive():
            # Shutdown uvicorn not in main thread
            # https://stackoverflow.com/questions/58010119/are-there-any-better-ways-to-run-uvicorn-in-thread
            self._uvicorn_server.should_exit = True

            run_async(
                asyncio.wait_for,
                self._uvicorn_server.shutdown(),
                shutdown_timeout or self.conf.timeout_for_shutdown_daemon
            )
            self._daemon_thread.join(shutdown_timeout or self.conf.timeout_for_shutdown_daemon)

    def run(self, daemon: bool = False):
        """
        Run web app with ML model.

        :param bool daemon: Run web app with daemon mode and unblock terminal. Default is False.

        """
        if not self.ml.loaded:
            raise WebAppLoadError(
                'ML Model not loaded to memory. For run web app, please call ml.load(), or ml.load().'
            )

        if not self.loaded:
            raise WebAppLoadError('For run web app, please call web.load(), or web.load().')

        if daemon is not True:
            self._run(in_thread=False)
            return

        if self._daemon_thread and self._daemon_thread.is_alive():
            logger.error(f'WebApp is already running. Thread name {self._daemon_thread.name}')
            raise WebAppLoadError(f'WebApp is already running. Thread name {self._daemon_thread.name}')

        self._daemon_thread = threading.Thread(
            target=self._run,
            kwargs={'in_thread': True},
            daemon=False,
            name='MLupWebAppDaemonThread'
        )
        self._daemon_thread.start()
        logger.info('Waiting start uvicorn proc with web app 30.0 seconds.')
        time.sleep(0.1)
        time_run = time.monotonic()
        while not self._uvicorn_server and time.monotonic() - time_run < 30.0:
            time.sleep(0.1)
        while self._uvicorn_server.started is False and time.monotonic() - time_run < 30.0:
            time.sleep(0.1)

    async def http_health(self):
        return {'status': 200}

    async def debug_info(self):
        info = await self.info()
        info.update({
            "model_config": self.ml.conf.ml_dict(),
            "web_app_config": self.conf.wb_dict(),
        })
        return info

    async def info(self):
        return {
            "model_info": {
                "name": self.ml.conf.name,
                "version": self.ml.conf.version,
                "type": self.ml.conf.type,
                "columns": self.ml.conf.columns,
            },
            "web_app_info": {
                "version": self.conf.web_app_version,
                "column_validation": self.conf.column_validation,
                "debug": self.conf.debug,
            },
        }

    @_set_predict_id_to_response_headers
    @_requests_throttling
    async def predict(self, request: FastAPIRequest, response: FastAPIResponse):
        predict_id = response.headers[PREDICT_ID_HEADER]
        # Validation
        try:
            predict_request_body = self._predict_inner_pydantic_model(**(await request.json()))
        except json.JSONDecodeError as e:
            raise PredictValidationInnerDataError(msg=f'Invalid json data: {e}', predict_id=predict_id)
        data_for_predict = predict_request_body.dict()

        # Throttling
        if self.conf.throttling_max_request_len:
            self._request_len_throttling(data_for_predict)

        # Predict in web app architecture object
        predict_result = await self._architecture_obj.predict(data_for_predict, predict_id=predict_id)

        # Result
        return {"predict_result": predict_result}
