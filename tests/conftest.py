import logging
from dataclasses import dataclass
import pickle
import time
from typing import List

import joblib
import pytest


logger = logging.getLogger('MLupTests')


class TestModel:
    def run_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        pass

    def predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        return self.run_predict(X, test_param, *custom_args, **custom_kwargs)

    def second_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        return self.run_predict(X, test_param, *custom_args, **custom_kwargs)

    def predict_with_x_name_y(self, Y: List, test_param: bool = False, *custom_args, **custom_kwargs):
        return self.run_predict(Y, test_param, *custom_args, **custom_kwargs)


class PrintModel(TestModel):
    def run_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        print(
            f'Call PrintModel.run_predict('
            f'X={X}, test_param={test_param}, *custom_args={custom_args}, custom_kwargs={custom_kwargs}'
            f')'
        )
        return X


@dataclass
class PrintSleepModel(TestModel):
    sleep: float = 1

    def run_predict(self, X: List, test_param: bool = False, *custom_args, **custom_kwargs):
        time.sleep(self.sleep)
        print(
            f'Call PrintModel.run_predict('
            f'X={X}, test_param={test_param}, *custom_args={custom_args}, custom_kwargs={custom_kwargs}'
            f')'
        )
        return X


@pytest.fixture(scope="session")
def pickle_print_model(tmp_path_factory):
    model = PrintModel()
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'pickle_print_model.pckl'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        pickle.dump(model, f)
    return f_path / model_name


@pytest.fixture(scope="session")
def pickle_print_sleep_model(tmp_path_factory):
    model = PrintSleepModel(sleep=1)
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'pickle_print_sleep_model.pckl'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        pickle.dump(model, f)
    return f_path / model_name


@pytest.fixture(scope="session")
def joblib_print_model(tmp_path_factory):
    model = PrintModel()
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'joblib_print_model.joblib'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        joblib.dump(model, f)
    return f_path / model_name


@pytest.fixture(scope="session")
def joblib_print_sleep_model(tmp_path_factory):
    model = PrintSleepModel(sleep=1)
    f_path = tmp_path_factory.getbasetemp()
    model_name = 'joblib_print_sleep_model.joblib'
    logger.info(f'Create {f_path}/{model_name} fixture.')
    with open(f_path / model_name, 'wb') as f:
        joblib.dump(model, f)
    return f_path / model_name


@pytest.fixture(scope="session")
def pickle_not_exists_file(tmp_path_factory):
    return tmp_path_factory.getbasetemp() / 'not_exists_file.pckl'
