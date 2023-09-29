import logging
import tempfile
from typing import Union, List

import onnx
import onnxruntime

from mlup.errors import ModelBinarizationError
from mlup.ml.binarization.base import BaseBinarizer
from mlup.constants import LoadedFile
from mlup.utils.profiling import TimeProfiler


logger = logging.getLogger('mlup')


class _InferenceSessionWithPredict(onnxruntime.InferenceSession):
    def format_predict(self, predict_result: List):
        if len(predict_result) > 1:
            return predict_result[:-1]
        return predict_result

    def predict(self, input_data):
        input_name = self.get_inputs()[0].name
        res = self.run(None, {input_name: input_data})
        return self.format_predict(res)


class _InferenceSessionWithPredictFullReturn(_InferenceSessionWithPredict):
    def format_predict(self, predict_result: List):
        return predict_result


class InferenceSessionBinarizer(BaseBinarizer):
    inference_class = _InferenceSessionWithPredict

    @classmethod
    def deserialize(cls, data: LoadedFile):
        logger.info('Run deserialization onnxruntime data.')
        with TimeProfiler('Time to deserialization onnxruntime data:'):
            try:
                _data = data.raw_data
                if data.path:
                    _data = data.path
                elif not isinstance(data.raw_data, str):
                    f = tempfile.NamedTemporaryFile(delete=True, suffix='.keras')
                    if isinstance(data.raw_data, bytes):
                        f.write(data.raw_data)
                    else:
                        f.write(data.raw_data.read())
                    f.seek(0)
                    data.path = f.name
                    _data = f.name
                return cls.inference_class(str(_data))
            except Exception as e:
                raise ModelBinarizationError(f'Error with deserialize model: {e}')

    @classmethod
    def _check_type(cls, model_data: Union[str, bytes]) -> bool:
        try:
            onnx.checker.check_model(model_data)
            return True
        except onnx.checker.ValidationError:
            return False
        except Exception:
            return False

    @classmethod
    def is_this_type(cls, loaded_file: LoadedFile) -> float:
        probability = 0.0
        if LoadedFile.raw_data:
            if cls._check_type(loaded_file.raw_data):
                probability = 0.9
        if loaded_file.path:
            if cls._check_type(loaded_file.path):
                probability = 0.9
            if str(loaded_file.path).endswith('.onnx'):
                probability += 0.05
        return probability


class InferenceSessionFullReturnBinarizer(InferenceSessionBinarizer):
    inference_class = _InferenceSessionWithPredictFullReturn
