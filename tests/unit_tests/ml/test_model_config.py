from mlup.constants import ModelLibraryType, StorageType, ModelDataTransformerType
from mlup.ml.model import ModelConfig


def test_model_config_default_values():
    conf = ModelConfig()
    assert conf.name == 'MyFirstMLupModel'
    assert conf.version == '1.0.0.0'
    assert conf.type == ModelLibraryType.SKLEARN
    assert conf.columns is None
    assert conf.predict_method_name == 'predict'
    assert conf.auto_detect_predict_params is True
    assert conf.storage_type == StorageType.memory
    assert conf.storage_kwargs == {}
    assert conf.binarization_type == 'auto'
    assert conf.use_thread_loop is True
    assert conf.max_thread_loop_workers is None
    assert conf.data_transformer_for_predict == ModelDataTransformerType.NUMPY_ARR
    assert conf.data_transformer_for_predicted == ModelDataTransformerType.NUMPY_ARR
