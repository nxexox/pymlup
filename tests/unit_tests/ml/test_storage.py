import logging
import os
import pickle
import shutil

import pytest

from mlup.constants import StorageType
from mlup.ml.storage.memory import MemoryStorage
from mlup.ml.storage.local_disk import DiskStorage
from mlup.utils.interspection import get_class_by_path


logger = logging.getLogger('mlup.test')


@pytest.fixture(scope="session")
def pickle_with_x_model(model_with_x_class, tmp_path_factory):
    f_path = tmp_path_factory.getbasetemp() / 'model_with_x'
    os.makedirs(f_path, exist_ok=True)
    model_name = 'pickle_easy_model_for_test_storage.pckl'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        pickle.dump(model_with_x_class(), f)
    return f_path / model_name


@pytest.fixture(scope="session")
def pickle_with_x_model_two(model_with_x_class, tmp_path_factory):
    f_path = tmp_path_factory.getbasetemp() / 'model_with_x'
    os.makedirs(f_path, exist_ok=True)
    model_name = 'pickle_easy_model_for_test_storage_two.pckl'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        pickle.dump(model_with_x_class(), f)
    return f_path / model_name


@pytest.mark.parametrize(
    'storage_type, storage_class', [
        (StorageType.memory, MemoryStorage),
        (StorageType.memory.value, MemoryStorage),
        (StorageType.disk, DiskStorage),
        (StorageType.disk.value, DiskStorage),
        ('mlup.constants.StorageType', StorageType),
    ]
)
def test_get_storage_class(storage_type, storage_class):
    assert get_class_by_path(storage_type) == storage_class


def test_get_storage_class_bad_storage_type():
    try:
        get_class_by_path('not exists storage type')
        pytest.fail('Not raised KeyError with not exists type')
    except ModuleNotFoundError as e:
        assert str(e) == str("No module named 'not exists storage type'")


class TestMemoryStorage:
    def test_load(self, model_with_x):
        storage = MemoryStorage(model=model_with_x)
        pred_models = storage.load()
        assert len(pred_models) == 1
        assert pred_models[0].raw_data == model_with_x

    def test_load_bytes_single_file(self, model_with_x):
        storage = MemoryStorage(model=model_with_x)
        pred_model = storage.load_bytes_single_file()
        assert pred_model == model_with_x


class TestDiskStorage:
    def test_load_from_single_file(self, pickle_with_x_model):
        storage = DiskStorage(path_to_files=pickle_with_x_model, need_load_file=True)
        models_bin = storage.load()
        assert len(models_bin) == 1
        with open(pickle_with_x_model, 'rb') as f:
            src_data = f.read()
        assert str(models_bin[0].path) == str(pickle_with_x_model)
        assert models_bin[0].raw_data == src_data

    def test_load_single_from_folder(self, pickle_with_x_model, joblib_print_model):
        # joblib_print_model for second file in folder
        storage = DiskStorage(path_to_files=os.path.dirname(pickle_with_x_model), need_load_file=True)
        models_bin = storage.load()
        assert len(models_bin) == 1
        with open(pickle_with_x_model, 'rb') as f:
            src_data = f.read()
        assert str(models_bin[0].path) == str(pickle_with_x_model)
        assert models_bin[0].raw_data == src_data

    def test_load_many_from_folder(self, pickle_with_x_model, pickle_with_x_model_two):
        storage = DiskStorage(path_to_files=os.path.dirname(pickle_with_x_model), need_load_file=True)
        models_bin = storage.load()
        assert len(models_bin) == 2
        with open(pickle_with_x_model, 'rb') as f:
            with_x_model_data = f.read()
        with open(pickle_with_x_model_two, 'rb') as f:
            with_x_model_data_two = f.read()
        # Order in folder
        assert str(models_bin[0].path) in {str(pickle_with_x_model), str(pickle_with_x_model_two)}
        assert models_bin[0].raw_data in {with_x_model_data, with_x_model_data_two}
        assert str(models_bin[1].path) == {str(pickle_with_x_model), str(pickle_with_x_model_two)}
        assert models_bin[1].raw_data == {with_x_model_data, with_x_model_data_two}

    def test_load_single_file_by_not_default_mask(self, joblib_print_model):
        storage = DiskStorage(
            path_to_files=joblib_print_model,
            files_mask=r'(\w.-_)*.joblib',
            need_load_file=True,
        )
        models_bin = storage.load()
        assert len(models_bin) == 1
        with open(joblib_print_model, 'rb') as f:
            src_data = f.read()
        assert str(models_bin[0].path) == str(joblib_print_model)
        assert models_bin[0].raw_data == src_data

    def test_load_many_file_by_not_default_mask(
        self, pickle_with_x_model, pickle_print_model, joblib_print_model, joblib_print_sleep_model
    ):
        path_to_folder = os.path.join(os.path.dirname(joblib_print_model), 'test_load_many_file_by_not_default_mask')
        os.makedirs(path_to_folder)
        joblib_print_model_new_path = os.path.join(path_to_folder, os.path.basename(joblib_print_model))
        joblib_print_sleep_model_new_path = os.path.join(path_to_folder, os.path.basename(joblib_print_sleep_model))
        shutil.copyfile(joblib_print_model, joblib_print_model_new_path)
        shutil.copyfile(joblib_print_sleep_model, joblib_print_sleep_model_new_path)
        shutil.copyfile(pickle_with_x_model, os.path.join(path_to_folder, os.path.basename(pickle_with_x_model)))
        shutil.copyfile(pickle_print_model, os.path.join(path_to_folder, os.path.basename(pickle_print_model)))

        storage = DiskStorage(
            path_to_files=path_to_folder,
            files_mask=r'(\w.-_)*.joblib',
            need_load_file=True,
        )
        models_bin = storage.load()
        assert len(models_bin) == 2
        with open(joblib_print_model_new_path, 'rb') as f:
            print_model_data = f.read()
        with open(joblib_print_sleep_model_new_path, 'rb') as f:
            print_sleep_model_data = f.read()
        # Order in folder
        for m in models_bin:
            assert str(m.path) in (str(joblib_print_model_new_path), str(joblib_print_sleep_model_new_path))
            assert m.raw_data in (print_model_data, print_sleep_model_data)

    # Folder, file
    @pytest.mark.parametrize(
        'model_path',
        ['pickle_not_exists_file', 'pickle_not_exists_folder'],
        ids=['file', 'folder']
    )
    def test_load_from_not_exists_path(self, model_path, request):
        model_path = request.getfixturevalue(model_path)
        storage = DiskStorage(path_to_files=model_path, need_load_file=True)
        try:
            storage.load()
            pytest.fail('Not raise ModelLoadError.')
        except FileNotFoundError as e:
            assert str(e) == f"[Errno 2] No such file or directory: '{model_path}'"

    def test_load_bytes_single_file(self, pickle_print_model):
        storage = DiskStorage(path_to_files='', need_load_file=True)
        model_bin = storage.load_bytes_single_file(path_to_file=pickle_print_model)
        with open(pickle_print_model, 'rb') as f:
            src_data = f.read()
        assert str(model_bin.path) == str(pickle_print_model)
        assert model_bin.raw_data == src_data

    def test_load_bytes_single_file_not_exists_file(self, pickle_not_exists_file):
        storage = DiskStorage(path_to_files='', need_load_file=True)
        try:
            storage.load_bytes_single_file(path_to_file=pickle_not_exists_file)
        except FileNotFoundError as e:
            assert str(e) == f"[Errno 2] No such file or directory: '{pickle_not_exists_file}'"
