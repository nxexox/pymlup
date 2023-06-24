from typing import Optional, List, Dict, Type, Union, Any

from fastapi import FastAPI
from pydantic import BaseModel as PydanticBaseModel

from mlup.constants import ModelDataType, ModelLibraryType, BinarizationType
from mlup.ml_model.data_transformers import BaseDataTransformer
from mlup.web_app.api_collections import ITEM_ID_COL_NAME, WebAppArchitecture


# TODO: For current work annotations and mypy
class MLupModelInterface:
    """Interface MLup model for current work annotations in MLup library."""
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
    config: 'mlup.config.MLupConfig' = None
    x_column_name: str = None

    @property
    def ml_model_obj(self):
        pass

    @property
    def web_app(self) -> 'MLupWebApp':
        pass

    def load(
        self,
        path_to_model_binary_files: Optional[str] = None,
        ml_model: Optional[Any] = None,
        force_loading: bool = False,
    ):
        pass

    def sync_predict(self, **predict_data):
        pass

    async def predict(self, **predict_data):
        pass

    def run_web_app(self, daemon: bool = False):
        pass

    def stop_web_app(self):
        pass

    def get_X_from_predict_data(self, predict_data: Dict, remove: bool = True) -> Optional[List[Union[Dict, List]]]:
        pass


class MLupWebAppInterface:
    """Interface MLup model for current work annotations in MLup library."""
    mlup_model: MLupModelInterface

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

    _web_app: FastAPI = None

    @property
    def web_app(self) -> FastAPI:
        pass

    def load(self):
        pass

    def stop_app(self):
        pass

    def run_app(self, daemon: bool = False):
        pass

