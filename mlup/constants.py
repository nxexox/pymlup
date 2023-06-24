from enum import Enum


IS_X = 'is_X'
DEFAULT_X_ARG_NAME = 'data_for_predict'


class BinarizationType(Enum):
    PICKLE = 'pickle'
    JOBLIB = 'joblib'


class ModelLibraryType(Enum):
    SKLEARN = 'sklearn'
    SCIKIT_LEARN = 'scikit-learn'


class ModelDataType(Enum):
    PANDAS_DF = 'DataFrame'
    NUMPY_ARR = 'NumpyArray'
    PYTHON_ARR = 'PythonArray'
    PYTHON_ARR_DICTS = 'PythonArrayDicts'
