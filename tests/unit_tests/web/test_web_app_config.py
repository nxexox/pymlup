from mlup.constants import ITEM_ID_COL_NAME, WebAppArchitecture
from mlup.web.app import WebAppConfig


def test_model_config_default_values():
    conf = WebAppConfig()
    assert conf.host == '0.0.0.0'
    assert conf.port == 8009
    assert conf.web_app_version == '1.0.0.0'
    assert conf.column_validation is False
    assert conf.custom_column_pydantic_model is None
    assert conf.mode == WebAppArchitecture.directly_to_predict
    assert conf.max_queue_size == 100
    assert conf.ttl_predicted_data == 60
    assert conf.ttl_client_wait == 30.0
    assert conf.min_batch_len == 10
    assert conf.batch_worker_timeout == 1.0
    assert conf.is_long_predict is False
    assert conf.throttling_max_requests is None
    assert conf.throttling_max_request_len is None
    assert conf.timeout_for_shutdown_daemon == 3.0
    assert conf.item_id_col_name == ITEM_ID_COL_NAME
