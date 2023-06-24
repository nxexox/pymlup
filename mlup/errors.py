class MLupError(Exception):
    pass


class ModelLoadError(MLupError):
    pass


class TransformDataError(MLupError):
    pass


class ModelPredictError(MLupError):
    pass
