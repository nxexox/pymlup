from dataclasses import dataclass


class MLupError(Exception):
    pass


class LoadError(MLupError):
    pass


class ModelLoadError(LoadError):
    pass


class WebAppLoadError(LoadError):
    pass


class ModelBinarizationError(ModelLoadError):
    pass


class PredictError(MLupError):
    http_status: int = 500
    type: str = 'predict_error'


@dataclass
class PredictWaitResultError(PredictError):
    msg: str
    predict_id: str
    http_status: int = 408
    type: str = 'predict_wait_result_error'

    def __str__(self):
        return self.msg


class PredictTransformDataError(PredictError):
    type: str = 'predict_transform_data_error'


@dataclass
class PredictValidationInnerDataError(PredictError):
    msg: str
    predict_id: str
    http_status: int = 422
    type: str = 'validation_error'

    def __str__(self):
        return self.msg
