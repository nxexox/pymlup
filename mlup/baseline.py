from dataclasses import dataclass
from typing import Union, Optional

from mlup.constants import ModelLibraryType, ModelDataTransformerType, BinarizationType, WebAppArchitecture
from mlup.up import Config


@dataclass
class WorkerAndQueueConfig(Config):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class BatchingConfig(Config):
    mode: WebAppArchitecture = WebAppArchitecture.batching


@dataclass
class ScikitLearnConfig(Config):
    type: Union[str, ModelLibraryType] = ModelLibraryType.SCIKIT_LEARN
    predict_method_name: str = 'predict'


@dataclass
class ScikitLearnWorkerConfig(ScikitLearnConfig):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class ScikitLearnBatchingConfig(ScikitLearnConfig):
    mode: WebAppArchitecture = WebAppArchitecture.batching


@dataclass
class TensorflowConfig(Config):
    type: Union[str, ModelLibraryType] = ModelLibraryType.TENSORFLOW
    predict_method_name: str = '__call__'
    data_transformer_for_predict: Union[str, ModelDataTransformerType] = ModelDataTransformerType.TENSORFLOW_TENSOR
    data_transformer_for_predicted: Union[str, ModelDataTransformerType] = ModelDataTransformerType.TENSORFLOW_TENSOR
    dtype_for_predict: Optional[str] = 'float32'


@dataclass
class TensorflowWorkerConfig(TensorflowConfig):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class TensorflowBatchingConfig(TensorflowConfig):
    mode: WebAppArchitecture = WebAppArchitecture.batching


@dataclass
class TorchConfig(Config):
    type: Union[str, ModelLibraryType] = ModelLibraryType.TORCH
    predict_method_name: str = '__call__'
    data_transformer_for_predict: Union[str, ModelDataTransformerType] = ModelDataTransformerType.TORCH_TENSOR
    data_transformer_for_predicted: Union[str, ModelDataTransformerType] = ModelDataTransformerType.TORCH_TENSOR
    dtype_for_predict: Optional[str] = 'float32'


@dataclass
class TorchWorkerConfig(TorchConfig):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class TorchBatchingConfig(TorchConfig):
    mode: WebAppArchitecture = WebAppArchitecture.batching


@dataclass
class OnnxConfig(Config):
    type: Union[str, ModelLibraryType] = ModelLibraryType.ONNX
    binarization_type: Union[str, BinarizationType] = BinarizationType.ONNX_INFERENCE_SESSION


@dataclass
class OnnxWorkerConfig(OnnxConfig):
    mode: WebAppArchitecture = WebAppArchitecture.worker_and_queue


@dataclass
class OnnxBatchingConfig(OnnxConfig):
    mode: WebAppArchitecture = WebAppArchitecture.batching
