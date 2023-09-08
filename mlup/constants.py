from enum import Enum
from io import BufferedIOBase
from typing import NamedTuple, Union, Callable, Optional

IS_X = 'is_X'
THERE_IS_ARGS = 'there_is_args'
DEFAULT_X_ARG_NAME = 'data_for_predict'
ITEM_ID_COL_NAME: str = 'mlup_item_id'
PREDICT_ID_HEADER = 'X-Predict-id'


class LoadedFile(NamedTuple):
    raw_data: Union[str, bytes, BufferedIOBase, Callable] = b''
    path: Optional[str] = None


class ModelLibraryType(Enum):
    SKLEARN = 'sklearn'
    SCIKIT_LEARN = 'scikit-learn'
    LIGHTGMB = 'lightgbm'
    TENSORFLOW = 'tensorflow'
    TORCH = 'torch'
    ONNX = 'onnx_inference_session'
    other = 'other'


class StorageType(Enum):
    memory = 'mlup.ml.storage.memory.MemoryStorage'
    disk = 'mlup.ml.storage.local_disk.DiskStorage'


class BinarizationType(Enum):
    MEMORY = 'mlup.ml.binarization.memory.MemoryBinarizer'
    JOBLIB = 'mlup.ml.binarization.joblib.JoblibBinarizer'
    PICKLE = 'mlup.ml.binarization.pickle.PickleBinarizer'
    LIGHTGBM = 'mlup.ml.binarization.lightgbm.LightGBMBinarizer'
    TENSORFLOW = 'mlup.ml.binarization.tensorflow.TensorFlowBinarizer'
    TENSORFLOW_ZIP = 'mlup.ml.binarization.tensorflow.TensorFlowSavedBinarizer'
    TORCH = 'mlup.ml.binarization.torch.TorchBinarizer'
    TORCH_JIT = 'mlup.ml.binarization.torch.TorchJITBinarizer'
    ONNX_INFERENCE_SESSION = 'mlup.ml.binarization.onnx.InferenceSessionBinarizer'


class ModelDataTransformerType(Enum):
    PANDAS_DF = 'mlup.ml.data_transformers.pandas_data_transformer.PandasDataFrameTransformer'
    NUMPY_ARR = 'mlup.ml.data_transformers.numpy_data_transformer.NumpyDataTransformer'
    TENSORFLOW_TENSOR = 'mlup.ml.data_transformers.tf_tensor_data_transformer.TFTensorDataTransformer'
    TORCH_TENSOR = 'mlup.ml.data_transformers.torch_tensor_data_transformer.TorchTensorDataTransformer'
    SRC_TYPES = 'mlup.ml.data_transformers.src_data_transformer.SrcDataTransformer'


class WebAppArchitecture(Enum):
    directly_to_predict = 'mlup.web.architecture.directly_to_predict.DirectlyToPredictArchitecture'
    worker_and_queue = 'mlup.web.architecture.worker_and_queue.WorkerAndQueueArchitecture'
    batching = 'mlup.web.architecture.batching.BatchingSingleProcessArchitecture'
