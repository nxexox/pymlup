import copy
from typing import List, Any

import pytest
from pydantic import BaseModel as PydanticBaseModel, create_model, ValidationError

from mlup.constants import IS_X, DEFAULT_X_ARG_NAME
from mlup.ml.model import MLupModel, ModelConfig
from mlup.web.api_validators import (
    make_map_pydantic_columns,
    create_pydantic_predict_model,
)


column_types_map = {
    'int': int,
    'float': float,
    'bool': bool,
    'str': str,
    'list': list,
}
src_columns = [
    {"name": "Float", "type": "float"},
    {"name": "FloatDefault", "type": "float", "default": 1.4},
    {"name": "FloatRequired", "type": "float", "required": True},
    {"name": "FloatNotRequired", "type": "float", "required": False},
    {"name": "FloatNotRequiredDefault", "type": "float", "required": False, "default": 1.4},
    {"name": "FloatRequiredDefault", "type": "float", "required": True, "default": 1.4},

    {"name": "Int", "type": "int"},
    {"name": "IntDefault", "type": "int", "default": 4},
    {"name": "IntRequired", "type": "int", "required": True},
    {"name": "IntNotRequired", "type": "int", "required": False},
    {"name": "IntNotRequiredDefault", "type": "int", "required": False, "default": 4},
    {"name": "IntRequiredDefault", "type": "int", "required": True, "default": 4},

    {"name": "Str", "type": "str"},
    {"name": "StrDefault", "type": "str", "default": "str"},
    {"name": "StrRequired", "type": "str", "required": True},
    {"name": "StrNotRequired", "type": "str", "required": False},
    {"name": "StrNotRequiredDefault", "type": "str", "required": False, "default": "str"},
    {"name": "StrRequiredDefault", "type": "str", "required": True, "default": "str"},

    {"name": "Bool", "type": "bool"},
    {"name": "BoolDefault", "type": "bool", "default": True},
    {"name": "BoolRequired", "type": "bool", "required": True},
    {"name": "BoolNotRequired", "type": "bool", "required": False},
    {"name": "BoolNotRequiredDefault", "type": "bool", "required": False, "default": True},
    {"name": "BoolRequiredDefault", "type": "bool", "required": True, "default": True},
]


def test_make_map_pydantic_columns():
    cols_configs, validators = make_map_pydantic_columns(src_columns)

    for col_config in src_columns:
        pred_col_type, pred_field_info = cols_configs.pop(col_config["name"])

        assert pred_col_type is column_types_map[col_config["type"]]
        assert pred_field_info.title == col_config["name"]
        if 'default' in col_config:
            assert pred_field_info.default is col_config["default"]
        elif col_config.get('required', True) is False:
            assert pred_field_info.default is None
        else:
            assert pred_field_info.default is Ellipsis

    assert len(cols_configs) == 0
    assert len(validators) == 0


def test_make_map_pydantic_validation():
    cols_configs, validators = make_map_pydantic_columns(src_columns)

    for col_config in src_columns:
        pred_col_type, pred_field_info = cols_configs.pop(col_config["name"])

        _test_pydantic_model = create_model(
            "_TestPydanticModel",
            **{col_config["name"]: (pred_col_type, pred_field_info)},
        )

        # Check valid type
        _test_pydantic_model(**{col_config["name"]: pred_col_type(1)})
        # Check not valid type
        try:
            not_valid_value = list
            _test_pydantic_model(**{col_config["name"]: not_valid_value})
            pytest.fail('Not raise error')
        except ValidationError as e:
            msg_str = str(e.raw_errors[0].exc.msg_template)
            if pred_col_type is str:
                assert msg_str == 'str type expected'
            elif pred_col_type is bool:
                assert msg_str == 'value could not be parsed to a boolean'
            else:
                assert msg_str.startswith(f'value is not a valid {col_config["type"]}')

        # Check required
        if col_config.get("required", True):
            # Check valid value
            _test_pydantic_model(**{col_config["name"]: pred_col_type(1)})
            # Check none value
            try:
                _test_pydantic_model(**{col_config["name"]: None})
                pytest.fail('Not raise error')
            except ValidationError as e:
                assert str(e.raw_errors[0].exc).startswith('none is not an allowed value')
        # Check not required
        else:
            # Check not exists value
            _test_pydantic_model()
            # Check None value
            try:
                _test_pydantic_model(**{col_config["name"]: None})
            except ValidationError as e:
                assert str(e.raw_errors[0].exc.msg_template) == 'none is not an allowed value'

    assert len(cols_configs) == 0
    assert len(validators) == 0


@pytest.mark.parametrize(
    'model_for_columns',
    [None, create_model("_TestPydanticModel")],
    ids=['WITHOUT_COLUMNS_MODEL', 'WITH_COLUMNS_MODEL']
)
def test_make_map_pydantic_columns_with_IS_X(model_for_columns):
    test_columns = copy.deepcopy(src_columns)
    test_columns[0][IS_X] = True

    _kwargs = dict(src_columns=test_columns)
    if model_for_columns:
        _kwargs['x_model'] = model_for_columns

    cols_configs, validators = make_map_pydantic_columns(**_kwargs)

    for col_config in test_columns:
        pred_col_type, pred_field_info = cols_configs.pop(col_config["name"])
        col_is_X = IS_X in col_config

        if col_is_X:
            if model_for_columns:
                assert pred_col_type is List[model_for_columns]
            else:
                assert pred_col_type is List[Any]
        else:
            assert pred_col_type is column_types_map[col_config["type"]]
        assert pred_field_info.title == col_config["name"]
        if 'default' in col_config:
            assert pred_field_info.default is col_config["default"]
        elif col_config.get('required', True) is False:
            assert pred_field_info.default is None
        else:
            assert pred_field_info.default is Ellipsis

    assert len(cols_configs) == 0
    assert len(validators) == 0


@pytest.mark.parametrize(
    'auto_detect_predict_params, x_param_name',
    [(True, 'X'), (False, DEFAULT_X_ARG_NAME)],
    ids=['auto_detect_predict_params=True', 'auto_detect_predict_params=False']
)
@pytest.mark.parametrize(
    'column_validation, columns, data, expected_data',
    [
        (False, src_columns, [1, 2, 3], [1, 2, 3]),
        (
            True,
            [src_columns[0], src_columns[6], src_columns[12]],
            [{'Float': 1.0, 'Int': 1, 'Str': '1', 'NotExistsKey': 10}],
            [{'Float': 1.0, 'Int': 1, 'Str': '1'}]
        ),
    ],
    ids=['column_validation=False', 'column_validation=True']
)
def test_create_pydantic_predict_model_valid(
    model_with_x,
    auto_detect_predict_params: bool,
    x_param_name: str,
    column_validation: bool,
    columns: List,
    data: List,
    expected_data: List,
):
    ml = MLupModel(
        ml_model=model_with_x,
        conf=ModelConfig(
            auto_detect_predict_params=auto_detect_predict_params, columns=columns
        )
    )
    ml.load()
    pred_pydantic_model = create_pydantic_predict_model(ml, column_validation=column_validation)

    # Check valid value
    data_for_pred = {x_param_name: data}
    ddt = pred_pydantic_model(**data_for_pred)
    assert ddt.dict() == {x_param_name: expected_data}

    # Check empty value
    try:
        pred_pydantic_model(not_exists_key=data).dict()
        pytest.fail('Not raise error')
    except ValidationError as e:
        assert str(e) == f'1 validation error for MLupRequestPredictPydanticModel\n{x_param_name}\n  ' \
                         'field required (type=value_error.missing)'

    # Check not valid value
    try:
        pred_pydantic_model(**{x_param_name: 1}).dict()
        pytest.fail('Not raise error')
    except ValidationError as e:
        assert str(e) == f'1 validation error for MLupRequestPredictPydanticModel\n{x_param_name}\n  ' \
                         'value is not a valid list (type=type_error.list)'


@pytest.mark.parametrize(
    'auto_detect_predict_params, x_param_name',
    [(True, 'X'), (False, DEFAULT_X_ARG_NAME)],
    ids=['auto_detect_predict_params=True', 'auto_detect_predict_params=False']
)
@pytest.mark.parametrize(
    'data, expected_data',
    [
        ([{'test_column': 1}], [{'test_column': '1'}]),
        ([{'test_column': 1, 'NotExistsKey': 10}], [{'test_column': '1'}]),
    ],
    ids=['data={"test_column": 1}', 'data={"test_column": 1, "not_exists_key": 1}']
)
def test_create_pydantic_predict_model_custom_column_pydantic_model(
    model_with_x,
    auto_detect_predict_params: bool,
    x_param_name: str,
    data: List,
    expected_data: List,
):
    ml = MLupModel(
        ml_model=model_with_x,
        conf=ModelConfig(
            auto_detect_predict_params=auto_detect_predict_params, columns=src_columns
        )
    )
    ml.load()

    class TestPydanticModel(PydanticBaseModel):
        test_column: str

    pred_pydantic_model = create_pydantic_predict_model(
        ml, column_validation=False, custom_column_pydantic_model=TestPydanticModel
    )

    # Check valid value
    data_for_pred = {x_param_name: data}
    ddt = pred_pydantic_model(**data_for_pred)
    assert ddt.dict() == {x_param_name: expected_data}

    # Check empty value
    try:
        pred_pydantic_model(not_exists_key=data).dict()
        pytest.fail('Not raise error')
    except ValidationError as e:
        assert str(e) == f'1 validation error for MLupRequestPredictPydanticModel\n{x_param_name}\n  ' \
                         'field required (type=value_error.missing)'

    # Check not valid
    try:
        pred_pydantic_model(**{x_param_name: 1}).dict()
        pytest.fail('Not raised error')
    except ValidationError as e:
        assert str(e) == f'1 validation error for MLupRequestPredictPydanticModel\n{x_param_name}\n  ' \
                         'value is not a valid list (type=type_error.list)'
