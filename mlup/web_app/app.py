import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial, wraps
import logging
from threading import Thread
from typing import Type, Optional, Dict


from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel as PydanticBaseModel, ValidationError
import uvicorn

from mlup.interfaces import MLupModelInterface
from mlup.errors import ModelPredictError
from mlup.utils.oop import MetaSingleton
from mlup.web_app.api_docs import openapi_schema
from mlup.web_app.api_errors import (
    ApiErrorResponse,
    api_exception_handler, ApiRequestError,
)
from mlup.web_app.api_validators import create_pydantic_predict_model, MlModelPredictRequest
from mlup.web_app.architecture import get_architecture
from mlup.web_app.architecture.base import BaseWebAppArchitecture
from mlup.web_app.api_collections import ITEM_ID_COL_NAME, WebAppArchitecture

logger = logging.getLogger('MLup')


@dataclass(kw_only=True)
class MLupWebApp(metaclass=MetaSingleton):
    """This is class for web app.
    Create object MLup with your ML model, set your settings and run your web app.

    Args:
        # Information arguments
        column_validation (bool): Add column validation after run predict in model.
            If this param set is True, then columns should be set.
            If this param set is False, then column validation will not run after call predict method by model.
            Default is False.

        custom_column_pydantic_model (Type[PydanticBaseModel]): Full custom data processing and validator for model.

    """
    mlup_model: MLupModelInterface

    # WebApp web interface settings
    host: str = '0.0.0.0'
    port: int = 8009
    web_app_version: str = '1.0.0.0'

    column_validation: bool = False
    custom_column_pydantic_model: Optional[Type[PydanticBaseModel]] = None

    # WebApp work mode settings
    mode: WebAppArchitecture = WebAppArchitecture.directly_to_predict
    # Max queue size for waiting batching.
    max_queue_size: int = 100
    # Max time member predict result data. Need for clean if client not returned for predict result.
    ttl_predicted_data: float = 60
    # Min batch len for start predict.
    min_batch_len: int = 10
    # Max time for pending before run batching.
    batch_worker_timeout: float = 1
    # Added get-predict api method and add return predict_id to predict response.
    is_long_predict: bool = False

    # WebApp work settings
    # Max count simultaneous requests to web app
    throttling_max_requests: Optional[int] = None
    # Max count objects to predict in single request
    throttling_max_request_len: Optional[int] = None
    # Wait time for graceful shutdown web app
    timeout_for_shutdown_daemon: float = 3

    # Column name for unique item_id
    item_id_col_name: str = ITEM_ID_COL_NAME

    # WebApp inner settings
    _fast_api: FastAPI = field(init=False, default=None)
    _throttling_max_requests_current_count: int = field(init=False, default=0)
    _predict_inner_pydantic_model: Type[MlModelPredictRequest] = field(init=False)
    _daemon_thread: Thread = field(init=False, default=None)
    _uvicorn_server: uvicorn.Server = field(init=False)
    _architecture_obj: BaseWebAppArchitecture = field(init=False)

    def __del__(self):
        self.stop_app()

    @property
    def web_app(self) -> FastAPI:
        if self._fast_api is None:
            logger.error('web_app nor creating. Please call web_app.load().')
            raise ValueError('web_app nor creating. Please call web_app.load().')
        return self._fast_api

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        # Startup web app code
        await self._architecture_obj.run()
        yield
        # Shutdown web app code
        await self._architecture_obj.stop()

    def _request_throttling(self, func):
        @wraps(func)
        async def wrap(*args, **kwargs):
            if self._throttling_max_requests_current_count < self.throttling_max_requests:
                self._throttling_max_requests_current_count += 1
                try:
                    return await func(*args, **kwargs)
                finally:
                    self._throttling_max_requests_current_count -= 1
            else:
                raise ApiRequestError(
                    'Max requests in app. Please try again later.',
                    status_code=429,
                    type='throttling_error'
                )
        return wrap

    def _request_len_throttling(self, data_for_predict: Dict):
        x_for_predict = self.mlup_model.get_X_from_predict_data(data_for_predict, remove=False)
        if x_for_predict and len(x_for_predict):
            raise ApiRequestError(
                'The query exceeded the limit on the number of rows for the predict. Please downsize your request.',
                status_code=429,
                type='throttling_error'
            )

    def _create_app(self):
        self._fast_api = FastAPI(
            lifespan=self._lifespan,
            exception_handlers={
                RequestValidationError: api_exception_handler,
                ValidationError: api_exception_handler,
                ModelPredictError: api_exception_handler,
                ApiRequestError: api_exception_handler,
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
        )

        # Set requests throttling
        if self.throttling_max_requests:
            self.predict = self._request_throttling(self.predict)

        self.web_app.add_api_route("/info", self.info, methods=["GET"])
        self.web_app.add_api_route("/predict", self.predict, methods=["POST"])

        self._architecture_obj = get_architecture(self.mode)(
            fast_api=self._fast_api,
            mlup_model=self.mlup_model,
            item_id_col_name=self.item_id_col_name,
            # Worker settings
            max_queue_size=self.max_queue_size,
            ttl_predicted_data=self.ttl_predicted_data,
            # Batching settings
            min_batch_len=self.min_batch_len,
            batch_worker_timeout=self.batch_worker_timeout,
            # Wait result from worker settings
            is_long_predict=self.is_long_predict,
        )
        self._architecture_obj.load()

    def _run(self):
        self._uvicorn_server = uvicorn.Server(
            uvicorn.Config(self.web_app, host=self.host, port=self.port)
        )
        self._uvicorn_server.run()

    def load(self):
        logger.info('Load Web application')

        if self.throttling_max_requests is not None and self.throttling_max_requests < 1:
            raise ValueError(
                f'The param throttling_max_requests must be greater than 0. '
                f'Now it is {self.throttling_max_requests}.'
            )
        if self.throttling_max_request_len is not None and self.throttling_max_request_len < 1:
            raise ValueError(
                f'The param throttling_max_request_len must be greater than 0. '
                f'Now it is {self.throttling_max_request_len}'
            )

        self._create_app()

        self._predict_inner_pydantic_model = create_pydantic_predict_model(self.mlup_model, self.column_validation)
        self.web_app.openapi = partial(openapi_schema, app=self.web_app, mlup_model=self.mlup_model)

    def stop_app(self):
        if self._daemon_thread and self._daemon_thread.is_alive():
            asyncio.run(
                asyncio.wait_for(
                    self._uvicorn_server.shutdown(),
                    self.timeout_for_shutdown_daemon
                )
            )
            self._daemon_thread.join(0.1)

    def run_app(self, daemon: bool = False):
        """
        Run web app with ML model.

        :param bool daemon: Run web app with daemon mode and unblock terminal. Default is False.

        """
        if not self.web_app:
            logger.error(f'For run web app, please call web_app.load(), or mlup_model.web_app.load().')
            return

        if daemon is not True:
            self._run()
            return

        if self._daemon_thread and self._daemon_thread.is_alive():
            logger.error(f'WebApp is already running. Thread name {self._daemon_thread.name}')
            return

        self._daemon_thread = Thread(
            target=self._run,
            args=(),
            daemon=True,
            name='MLupWebAppDaemonThread'
        )
        self._daemon_thread.start()

    async def info(self):
        return {
            "model_info": {
                "name": self.mlup_model.name,
                "version": self.mlup_model.version,
                "type": self.mlup_model.type,
                "columns": self.mlup_model.columns,
            },
            "model_config": {
                "predict_method": self.mlup_model.predict_method_name,
                "auto_detect_predict_params": self.mlup_model.auto_detect_predict_params,
                "use_thread_loop": self.mlup_model.use_thread_loop,
                "max_thread_loop_workers": self.mlup_model.max_thread_loop_workers,
                "data_type_for_predict": self.mlup_model.model_data_type_for_predict,
                "data_type_for_predicted": self.mlup_model.model_data_type_for_predicted,
                "x_column_name": self.mlup_model.x_column_name,
            },
            "web_app_info": {
                "version": self.web_app_version,
                "column_validation": self.column_validation,
                "custom_column_pydantic_model": self.custom_column_pydantic_model,
            },
            "web_app_config": {},
        }

    async def predict(self, request: FastAPIRequest):
        # Validation
        predict_request_body = self._predict_inner_pydantic_model(**(await request.json()))
        data_for_predict = predict_request_body.dict()

        # Throttling
        if self.throttling_max_request_len:
            self._request_len_throttling(data_for_predict)

        # Predict in web app architecture object
        predict_result = await self._architecture_obj.predict(data_for_predict)

        # Result
        return {"predict_result": predict_result}
