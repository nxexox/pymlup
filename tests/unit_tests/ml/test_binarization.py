import logging
from io import BytesIO

import pytest

from mlup.constants import BinarizationType, LoadedFile
from mlup.ml.binarization.memory import MemoryBinarizer
from mlup.ml.binarization.pickle import PickleBinarizer
from mlup.utils.interspection import get_class_by_path

logger = logging.getLogger('mlup.test')
try:
    from mlup.ml.binarization.joblib import JoblibBinarizer
except (ModuleNotFoundError, AttributeError) as e:
    logger.info(f'joblib library not installed. Skip test. {e}')
    JoblibBinarizer = None
try:
    from mlup.ml.binarization.lightgbm import LightGBMBinarizer
except (ModuleNotFoundError, AttributeError) as e:
    logger.info(f'lightgbm library not installed. Skip test. {e}')
    LightGBMBinarizer = None
try:
    from mlup.ml.binarization.onnx import InferenceSessionBinarizer
except (ModuleNotFoundError, AttributeError) as e:
    logger.info(f'onnxruntime library not installed. Skip test. {e}')
    InferenceSessionBinarizer = None
try:
    from mlup.ml.binarization.tensorflow import TensorFlowBinarizer, TensorFlowSavedBinarizer
except (ModuleNotFoundError, AttributeError) as e:
    logger.info(f'tensorflow library not installed. Skip test. {e}')
    TensorFlowBinarizer, TensorFlowSavedBinarizer = None, None
try:
    from mlup.ml.binarization.torch import TorchBinarizer, TorchJITBinarizer
except (ModuleNotFoundError, AttributeError) as e:
    logger.info(f'pytorch library not installed. Skip test. {e}')
    TorchBinarizer, TorchJITBinarizer = None, None


@pytest.fixture(scope="session")
def model_not_readable_from_disk(tmp_path_factory):
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'model_not_readable_from_disk.bin'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'w') as f:
        f.write(model_name)
    return f_path / model_name


def test_single_file_memory_binarization_deserialize(print_model):
    binarizer = MemoryBinarizer
    model = binarizer.deserialize(LoadedFile(print_model))

    src_x = [1, 2, 3]
    pred_d = model.predict(src_x, True, 4, 5, 6, seven=7, eight=8)
    assert pred_d == src_x


def test_single_file_pickle_binarization_deserialize(pickle_print_model):
    binarizer = PickleBinarizer
    with open(pickle_print_model, 'rb') as f:
        data = f.read()

    model = binarizer.deserialize(LoadedFile(data, pickle_print_model))

    src_x = [1, 2, 3]
    pred_d = model.predict(src_x, True, 4, 5, 6, seven=7, eight=8)
    assert pred_d == src_x


@pytest.mark.parametrize(
    'data',
    [
        pytest.param(
            b'\x80\x04\x95$\x00\x00\x00\x00\x00\x00\x00\x8c\x0etests.conftest\x94\x8c\nPrintModel\x94\x93\x94)\x81\x94.',
            id='bytes'
        ),
        pytest.param(
            BytesIO(b'\x80\x04\x95$\x00\x00\x00\x00\x00\x00\x00\x8c\x0etests.conftest\x94\x8c\nPrintModel\x94\x93\x94)'
                    b'\x81\x94.'),
            id='BufferedReader'
        )
    ]
)
@pytest.mark.skipif(JoblibBinarizer is None, reason='joblib library not installed.')
def test_single_file_joblib_binarization_deserialize(data):
    binarizer = JoblibBinarizer

    model = binarizer.deserialize(LoadedFile(data))

    src_x = [1, 2, 3]
    pred_d = model.predict(src_x, True, 4, 5, 6, seven=7, eight=8)
    assert pred_d == src_x


@pytest.mark.skipif(InferenceSessionBinarizer is None, reason='sklearn, onnxruntime libraries not installed.')
def test_single_file_sklearn_onnx_binarization_deserialize(scikit_learn_binary_cls_model_onnx):
    import numpy as np
    binarizer = InferenceSessionBinarizer
    test_data_raw = np.array([list(scikit_learn_binary_cls_model_onnx.test_data_raw.values())], dtype=np.float32)

    with open(scikit_learn_binary_cls_model_onnx.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(f.read(), scikit_learn_binary_cls_model_onnx.path))
        pred_d = model.predict(test_data_raw)
        assert pred_d == scikit_learn_binary_cls_model_onnx.test_model_response_raw

    with open(scikit_learn_binary_cls_model_onnx.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(BytesIO(f.read()), scikit_learn_binary_cls_model_onnx.path))
        pred_d = model.predict(test_data_raw)
        assert pred_d == scikit_learn_binary_cls_model_onnx.test_model_response_raw


@pytest.mark.skipif(LightGBMBinarizer is None, reason='lightgbm library not installed.')
def test_single_file_lightgbm_binarization_deserialize(lightgbm_binary_cls_model_txt):
    import pandas
    import numpy as np
    binarizer = LightGBMBinarizer
    test_data_raw = pandas.DataFrame(
        [lightgbm_binary_cls_model_txt.test_data_raw],
        columns=list(lightgbm_binary_cls_model_txt.test_data_raw.keys())
    )

    with open(lightgbm_binary_cls_model_txt.path, 'r') as f:
        model = binarizer.deserialize(LoadedFile(f.read(), lightgbm_binary_cls_model_txt.path))
        pred_d = model.predict(test_data_raw)
        assert np.array_equal(
            pred_d,
            np.array([lightgbm_binary_cls_model_txt.test_model_response_raw])
        )

    with open(lightgbm_binary_cls_model_txt.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(BytesIO(f.read()), lightgbm_binary_cls_model_txt.path))
        pred_d = model.predict(test_data_raw)
        assert np.array_equal(
            pred_d,
            np.array([lightgbm_binary_cls_model_txt.test_model_response_raw])
        )


@pytest.mark.parametrize(
    'fixture_name',
    ['tensorflow_binary_cls_model_keras', 'tensorflow_binary_cls_model_h5']
)
@pytest.mark.skipif(TensorFlowBinarizer is None, reason='tensorflow library not installed.')
def test_single_file_tensorflow_binarization_deserialize(fixture_name, request):
    import tensorflow as tf
    binarizer = TensorFlowBinarizer
    base_model = request.getfixturevalue(fixture_name)
    test_data_raw = tf.convert_to_tensor([list(base_model.test_data_raw.values())])

    with open(base_model.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(f.read(), base_model.path))
        pred_d = model(test_data_raw)
        assert pred_d.numpy().tolist() == [[base_model.test_model_response_raw]]

    with open(base_model.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(BytesIO(f.read()), base_model.path))
        pred_d = model(test_data_raw)
        assert pred_d.numpy().tolist() == [[base_model.test_model_response_raw]]


@pytest.mark.skipif(TorchBinarizer is None, reason='pytorch library not installed.')
def test_single_file_torch_binarization_deserialize(pytorch_binary_cls_model_pth):
    import torch
    binarizer = TorchBinarizer
    test_data_raw = torch.tensor(list(pytorch_binary_cls_model_pth.test_data_raw.values()))

    with open(pytorch_binary_cls_model_pth.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(f.read(), pytorch_binary_cls_model_pth.path))
        pred_d = model(test_data_raw)
        assert pred_d == pytorch_binary_cls_model_pth.test_model_response_raw

    with open(pytorch_binary_cls_model_pth.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(BytesIO(f.read()), pytorch_binary_cls_model_pth.path))
        pred_d = model(test_data_raw)
        assert pred_d == pytorch_binary_cls_model_pth.test_model_response_raw


@pytest.mark.skipif(InferenceSessionBinarizer is None, reason='pytorch, onnxruntime libraries not installed.')
def test_single_file_torch_onnx_binarization_deserialize(pytorch_binary_cls_model_onnx):
    import numpy as np
    binarizer = InferenceSessionBinarizer
    test_data_raw = np.array([list(pytorch_binary_cls_model_onnx.test_data_raw.values())], dtype=np.float32)

    with open(pytorch_binary_cls_model_onnx.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(f.read(), pytorch_binary_cls_model_onnx.path))
        pred_d = model.predict(test_data_raw)
        assert pred_d[0].tolist()[0][0] == pytorch_binary_cls_model_onnx.test_model_response_raw

    with open(pytorch_binary_cls_model_onnx.path, 'rb') as f:
        model = binarizer.deserialize(LoadedFile(BytesIO(f.read()), pytorch_binary_cls_model_onnx.path))
        pred_d = model.predict(test_data_raw)
        assert pred_d[0].tolist()[0][0] == pytorch_binary_cls_model_onnx.test_model_response_raw


@pytest.mark.parametrize(
    'binarization_type, binarizer', [
        (BinarizationType.MEMORY, MemoryBinarizer),
        (BinarizationType.MEMORY.value, MemoryBinarizer),
        (BinarizationType.PICKLE, PickleBinarizer),
        (BinarizationType.PICKLE.value, PickleBinarizer),
        (BinarizationType.JOBLIB, JoblibBinarizer),
        (BinarizationType.JOBLIB.value, JoblibBinarizer),
        (BinarizationType.LIGHTGBM, LightGBMBinarizer),
        (BinarizationType.LIGHTGBM.value, LightGBMBinarizer),
        (BinarizationType.TENSORFLOW, TensorFlowBinarizer),
        (BinarizationType.TENSORFLOW.value, TensorFlowBinarizer),
        (BinarizationType.TENSORFLOW_ZIP, TensorFlowSavedBinarizer),
        (BinarizationType.TENSORFLOW_ZIP.value, TensorFlowSavedBinarizer),
        (BinarizationType.TORCH, TorchBinarizer),
        (BinarizationType.TORCH.value, TorchBinarizer),
        (BinarizationType.TORCH_JIT, TorchJITBinarizer),
        (BinarizationType.TORCH_JIT.value, TorchJITBinarizer),
        (BinarizationType.ONNX_INFERENCE_SESSION, InferenceSessionBinarizer),
        (BinarizationType.ONNX_INFERENCE_SESSION.value, InferenceSessionBinarizer),
        ('mlup.constants.BinarizationType', BinarizationType),
    ]
)
def test_get_binarizer(binarization_type, binarizer):
    if binarizer is None:
        return
    assert binarizer == get_class_by_path(binarization_type)


def test_get_binarizer_bad_binarization_type():
    try:
        get_class_by_path('not exists binarization type')
        pytest.fail('Not raised KeyError with not exists type')
    except ModuleNotFoundError as e:
        assert str(e) == str("No module named 'not exists binarization type'")
