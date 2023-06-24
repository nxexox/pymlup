import logging

import pytest

from mlup.constants import BinarizationType
from mlup.errors import ModelLoadError
from mlup.ml_model.binarization import SingleFileBinarization, model_load_binary, ModelLibraryType


logger = logging.getLogger('MLupTests')


@pytest.fixture(scope="session")
def model_not_readable_from_disk(tmp_path_factory):
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'model_not_readable_from_disk.bin'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'w') as f:
        f.write(model_name)
    return f_path / model_name


def test_single_file_binarization_deserialize_pickle(pickle_print_model):
    binarizer = SingleFileBinarization(pickle_print_model)
    model = binarizer.deserialize_pickle()

    src_x = [1, 2, 3]
    pred_d = model.predict(src_x, True, 4, 5, 6, seven=7, eight=8)
    assert src_x == pred_d


def test_single_file_binarization_deserialize_joblib(joblib_print_model):
    binarizer = SingleFileBinarization(joblib_print_model)
    model = binarizer.deserialize_joblib()

    src_x = [1, 2, 3]
    pred_d = model.predict(src_x, True, 4, 5, 6, seven=7, eight=8)
    assert src_x == pred_d


def test_single_file_binarization_deserialize_model(pickle_print_model):
    binarizer = SingleFileBinarization(pickle_print_model)
    model = binarizer.deserialize_model()

    src_x = [1, 2, 3]
    pred_d = model.predict(src_x, True, 4, 5, 6, seven=7, eight=8)
    assert src_x == pred_d


@pytest.mark.parametrize(
    'model_type', [
        ModelLibraryType.SKLEARN,
        ModelLibraryType.SCIKIT_LEARN,
        ModelLibraryType.SKLEARN.value,
        ModelLibraryType.SCIKIT_LEARN.value
    ]
)
def test_model_load_binary(pickle_print_model, joblib_print_model, model_type):
    src_x = [1, 2, 3]

    pickle_model = model_load_binary(model_type, pickle_print_model, BinarizationType.PICKLE)
    pred_d = pickle_model.predict(src_x, True, 4, 5, 6, seven=7, eight=8)
    assert src_x == pred_d

    joblib_model = model_load_binary(model_type, joblib_print_model, BinarizationType.JOBLIB)
    pred_d = joblib_model.predict(src_x, True, 4, 5, 6, seven=7, eight=8)
    assert src_x == pred_d


def test_model_load_binary_bad_model_type(pickle_print_model):
    try:
        model_load_binary('not exists model type', pickle_print_model)
        pytest.fail('Not raised KeyError with not exists type')
    except KeyError:
        pass


def test_model_load_binary_bad_deserialize(model_not_readable_from_disk):
    try:
        model_load_binary(
            ModelLibraryType.SKLEARN,
            model_not_readable_from_disk,
        )
        pytest.fail('Not raises ModelLoadError when deserialize pickle with joblib.')
    except ModelLoadError as e:
        assert str(e) == "Error with deserialize model: invalid load key, 'm'."
