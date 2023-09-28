import logging
import os
import pickle

import pytest

from mlup.up import UP, Config
from mlup.constants import ModelDataTransformerType, StorageType, BinarizationType, WebAppArchitecture
from mlup.ml.model import MLupModel, ModelConfig

logger = logging.getLogger('mlup.test')
try:
    import joblib
except (ModuleNotFoundError, AttributeError) as e:
    logger.info(f'joblib library not installed. Skip test. {e}')
    joblib = None


@pytest.mark.asyncio
async def test_mlup_model_pickle_serilization(pickle_print_model, models_datadir):
    mlup_model = MLupModel(
        conf=ModelConfig(
            data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            storage_type=StorageType.disk,
            storage_kwargs={'path_to_files': pickle_print_model},
        )
    )
    mlup_model.load()
    pred = await mlup_model.predict(X=[[1, 2, 3]])
    assert pred == [[1, 2, 3]]

    mlup_model_pickle_data = pickle.dumps(mlup_model)
    mlup_unpickle = pickle.loads(mlup_model_pickle_data)

    pred = await mlup_unpickle.predict(X=[[1, 2, 3]])
    assert pred == [[1, 2, 3]]


@pytest.mark.asyncio
@pytest.mark.skipif(joblib is None, reason='joblib library not installed.')
async def test_mlup_model_joblib_serilization(joblib_print_model):
    mlup_model = MLupModel(
        conf=ModelConfig(
            data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            storage_type=StorageType.disk,
            storage_kwargs={'path_to_files': joblib_print_model, 'files_mask': r'(\w.-_)*.joblib'},
            binarization_type=BinarizationType.JOBLIB,
        )
    )
    mlup_model.load()
    pred = await mlup_model.predict(X=[[1, 2, 3]])
    assert pred == [[1, 2, 3]]

    new_model_path = os.path.join(
        str(joblib_print_model).rsplit('/', 1)[0],
        'test_mlup_model_joblib_serilization'
    )
    new_model_name = os.path.join(new_model_path, 'joblib_model.joblib')
    os.makedirs(new_model_path)

    joblib.dump(mlup_model, new_model_name)
    mlup_unjoblib = joblib.load(new_model_name)

    pred = await mlup_unjoblib.predict(X=[[1, 2, 3]])
    assert pred == [[1, 2, 3]]


@pytest.mark.parametrize(
    'mode',
    [
        WebAppArchitecture.directly_to_predict,
        WebAppArchitecture.worker_and_queue,
        WebAppArchitecture.batching,
    ],
    ids=['web.mode=directly_to_predict', 'web.mode=worker_and_queue', 'web.mode=batching']
)
@pytest.mark.asyncio
async def test_mlup_pickle_serilization(pickle_print_model, mode: WebAppArchitecture):
    up = UP(
        conf=Config(
            data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            storage_type=StorageType.disk,
            storage_kwargs={'path_to_files': pickle_print_model},
            mode=mode,
        )
    )
    up.ml.load()
    up.web.load()
    pred = up.predict(X=[[1, 2, 3]])
    assert pred == [[1, 2, 3]]

    up_pickle_data = pickle.dumps(up)
    up_unpickle = pickle.loads(up_pickle_data)

    pred = up_unpickle.predict(X=[[1, 2, 3]])
    assert pred == [[1, 2, 3]]


@pytest.mark.parametrize(
    'mode',
    [
        WebAppArchitecture.directly_to_predict,
        WebAppArchitecture.worker_and_queue,
        WebAppArchitecture.batching,
    ],
    ids=['web.mode=directly_to_predict', 'web.mode=worker_and_queue', 'web.mode=batching']
)
@pytest.mark.asyncio
@pytest.mark.skipif(joblib is None, reason='joblib library not installed.')
async def test_mlup_joblib_serilization(joblib_print_model, mode: WebAppArchitecture):
    up = UP(
        conf=Config(
            data_transformer_for_predict=ModelDataTransformerType.NUMPY_ARR,
            storage_type=StorageType.disk,
            storage_kwargs={'path_to_files': joblib_print_model, 'files_mask': r'(\w.-_)*.joblib'},
            binarization_type=BinarizationType.JOBLIB,
            mode=mode,
        )
    )
    up.ml.load()
    up.web.load()
    pred = up.predict(X=[[1, 2, 3]])
    assert pred == [[1, 2, 3]]

    new_up_path = os.path.join(
        str(joblib_print_model).rsplit('/', 1)[0],
        'test_mlup_joblib_serilization'
    )
    new_up_name = os.path.join(new_up_path, 'joblib_up.joblib')
    if os.path.exists(new_up_path):
        new_up_name = new_up_path.rsplit('/', 1)[0] + \
                      new_up_path.rsplit('/', 1)[1] + \
                      f'{len(os.listdir(new_up_path))}_joblib_up.joblib'
    else:
        os.makedirs(new_up_path)

    joblib.dump(up, new_up_name)
    up_unjoblib = joblib.load(new_up_name)

    pred = up_unjoblib.predict(X=[[1, 2, 3]])
    assert pred == [[1, 2, 3]]
