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
    'List': List,
}
# Pydantic v2 ValidationError "type" taxonomy for each mlup column type (see `.errors()`,
# https://errors.pydantic.dev/2.13/migration/). Used to check errors structurally instead of
# comparing full Pydantic v1 error message text (`raw_errors`/`msg_template`), which don't
# exist in Pydantic v2.
error_type_by_col_type = {
    'int': 'int_type',
    'float': 'float_type',
    'bool': 'bool_type',
    'str': 'string_type',
}
src_columns = [
    {"name": "Float", "type": "float"},
    {"name": "FloatDefault", "type": "float", "default": 1.4},
    {"name": "FloatRequired", "type": "float", "required": True},
    {"name": "FloatNotRequired", "type": "float", "required": False},
    {"name": "FloatNotRequiredDefault", "type": "float", "required": False, "default": 1.4},
    {"name": "FloatRequiredDefault", "type": "float", "required": True, "default": 1.4},
    {"name": "FloatList", "type": "float", "collection_type": "List"},
    {"name": "FloatOptionalList", "type": "float", "collection_type": "List", "required": False},

    {"name": "Int", "type": "int"},
    {"name": "IntDefault", "type": "int", "default": 4},
    {"name": "IntRequired", "type": "int", "required": True},
    {"name": "IntNotRequired", "type": "int", "required": False},
    {"name": "IntNotRequiredDefault", "type": "int", "required": False, "default": 4},
    {"name": "IntRequiredDefault", "type": "int", "required": True, "default": 4},
    {"name": "IntList", "type": "int", "collection_type": "List"},
    {"name": "IntOptionalList", "type": "int", "collection_type": "List", "required": False},

    {"name": "Str", "type": "str"},
    {"name": "StrDefault", "type": "str", "default": "str"},
    {"name": "StrRequired", "type": "str", "required": True},
    {"name": "StrNotRequired", "type": "str", "required": False},
    {"name": "StrNotRequiredDefault", "type": "str", "required": False, "default": "str"},
    {"name": "StrRequiredDefault", "type": "str", "required": True, "default": "str"},
    {"name": "StrList", "type": "str", "collection_type": "List"},
    {"name": "StrOptionalList", "type": "str", "collection_type": "List", "required": False},

    {"name": "Bool", "type": "bool"},
    {"name": "BoolDefault", "type": "bool", "default": True},
    {"name": "BoolRequired", "type": "bool", "required": True},
    {"name": "BoolNotRequired", "type": "bool", "required": False},
    {"name": "BoolNotRequiredDefault", "type": "bool", "required": False, "default": True},
    {"name": "BoolRequiredDefault", "type": "bool", "required": True, "default": True},
    {"name": "BoolList", "type": "bool", "collection_type": "List"},
    {"name": "BoolOptionalList", "type": "bool", "collection_type": "List", "required": False},
]


def test_make_map_pydantic_columns():
    cols_configs, validators = make_map_pydantic_columns(src_columns)

    for col_config in src_columns:
        pred_col_type, pred_field_info = cols_configs.pop(col_config["name"])

        if "collection_type" in col_config:
            assert pred_col_type is column_types_map[col_config["collection_type"]][column_types_map[col_config["type"]]]
        else:
            assert pred_col_type is column_types_map[col_config["type"]]
        assert pred_field_info.title == col_config["name"]
        if 'default' in col_config:
            assert pred_field_info.default is col_config["default"]
            assert pred_field_info.is_required() is False
        elif col_config.get('required', True) is False:
            assert pred_field_info.default is None
            assert pred_field_info.is_required() is False
        else:
            assert pred_field_info.is_required() is True

    assert len(cols_configs) == 0
    assert len(validators) == 0


def test_make_map_pydantic_validation():
    cols_configs, validators = make_map_pydantic_columns(src_columns)

    for col_config in src_columns:
        pred_col_type, pred_field_info = cols_configs.pop(col_config["name"])
        expected_type_error = (
            'list_type' if "collection_type" in col_config else error_type_by_col_type[col_config["type"]]
        )

        _test_pydantic_model = create_model(
            "_TestPydanticModel",
            **{col_config["name"]: (pred_col_type, pred_field_info)},
        )

        # Check valid type
        if "collection_type" in col_config:
            _test_pydantic_model(**{col_config["name"]: [column_types_map[col_config["type"]](1)]})
        else:
            _test_pydantic_model(**{col_config["name"]: pred_col_type(1)})
        # Check not valid type
        try:
            not_valid_value = list
            _test_pydantic_model(**{col_config["name"]: not_valid_value})
            pytest.fail('Not raise error')
        except ValidationError as e:
            assert e.errors()[0]['type'] == expected_type_error

        # Check required
        if col_config.get("required", True):
            # Check valid value
            if "collection_type" in col_config:
                _test_pydantic_model(**{col_config["name"]: [column_types_map[col_config["type"]](1)]})
            else:
                _test_pydantic_model(**{col_config["name"]: pred_col_type(1)})
            # Check none value
            try:
                _test_pydantic_model(**{col_config["name"]: None})
                pytest.fail('Not raise error')
            except ValidationError as e:
                assert e.errors()[0]['type'] == expected_type_error
        # Check not required
        else:
            # Check not exists value: uses the default (None) as-is, without validation.
            _test_pydantic_model()
            # Check explicit None value: mlup's not-required columns only set `default=None`,
            # they don't widen the field type to `Optional[...]`. Pydantic v1 inferred implicit
            # optionality from `default=None` and accepted an explicit `null`; Pydantic v2 does
            # not do this anymore, so an explicit `null` is now rejected with the same
            # structural type error as any other wrong-type value. This is a real, documented
            # behavior change (see CHANGELOG), not a test artifact.
            try:
                _test_pydantic_model(**{col_config["name"]: None})
                pytest.fail('Not raise error')
            except ValidationError as e:
                assert e.errors()[0]['type'] == expected_type_error

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
        elif "collection_type" in col_config:
            assert pred_col_type is List[column_types_map[col_config["type"]]]
        else:
            assert pred_col_type is column_types_map[col_config["type"]]
        assert pred_field_info.title == col_config["name"]
        if 'default' in col_config:
            assert pred_field_info.default is col_config["default"]
            assert pred_field_info.is_required() is False
        elif col_config.get('required', True) is False:
            assert pred_field_info.default is None
            assert pred_field_info.is_required() is False
        else:
            assert pred_field_info.is_required() is True

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
        (False, src_columns, [1, [4, 5], 2, 3], [1, [4, 5], 2, 3]),
        (
            True,
            [src_columns[0], src_columns[6], src_columns[8], src_columns[16]],
            [{'Float': 1.0, 'FloatList': [1.0, 2.0], 'Int': 1, 'Str': '1', 'NotExistsKey': 10}],
            [{'Float': 1.0, 'FloatList': [1.0, 2.0], 'Int': 1, 'Str': '1'}]
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
    assert ddt.model_dump() == {x_param_name: expected_data}

    # Check empty value
    try:
        pred_pydantic_model(not_exists_key=data).model_dump()
        pytest.fail('Not raise error')
    except ValidationError as e:
        error = e.errors()[0]
        assert error['loc'] == (x_param_name,)
        assert error['type'] == 'missing'

    # Check not valid value
    try:
        pred_pydantic_model(**{x_param_name: 1}).model_dump()
        pytest.fail('Not raise error')
    except ValidationError as e:
        error = e.errors()[0]
        assert error['loc'] == (x_param_name,)
        assert error['type'] == 'list_type'


@pytest.mark.parametrize(
    'auto_detect_predict_params, x_param_name',
    [(True, 'X'), (False, DEFAULT_X_ARG_NAME)],
    ids=['auto_detect_predict_params=True', 'auto_detect_predict_params=False']
)
@pytest.mark.parametrize(
    'data, expected_data',
    [
        # Pydantic v1 silently coerced an int to str here (`1` -> `'1'`); Pydantic v2 no
        # longer coerces non-str input for a `str` field, so the input must already be a
        # string. Unknown keys (`NotExistsKey`) are still stripped by the default
        # `extra='ignore'` model behavior, unchanged between v1 and v2.
        ([{'test_column': '1'}], [{'test_column': '1'}]),
        ([{'test_column': '1', 'NotExistsKey': 10}], [{'test_column': '1'}]),
    ],
    ids=['data={"test_column": "1"}', 'data={"test_column": "1", "not_exists_key": 1}']
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
    assert ddt.model_dump() == {x_param_name: expected_data}

    # Check empty value
    try:
        pred_pydantic_model(not_exists_key=data).model_dump()
        pytest.fail('Not raise error')
    except ValidationError as e:
        error = e.errors()[0]
        assert error['loc'] == (x_param_name,)
        assert error['type'] == 'missing'

    # Check not valid
    try:
        pred_pydantic_model(**{x_param_name: 1}).model_dump()
        pytest.fail('Not raised error')
    except ValidationError as e:
        error = e.errors()[0]
        assert error['loc'] == (x_param_name,)
        assert error['type'] == 'list_type'


def test_create_pydantic_predict_model_rejects_pydantic_v1_custom_column_model(model_with_x):
    # Regression test: mlup only supports Pydantic v2 models as `custom_column_pydantic_model`
    # now. A model built on the `pydantic.v1` compatibility namespace (shipped inside Pydantic
    # v2 installs) must not be silently accepted - mlup doesn't add a `pydantic.v1` dependency
    # or a compatibility layer for it.
    import pydantic.v1

    class V1CustomColumnModel(pydantic.v1.BaseModel):
        test_column: str

    ml = MLupModel(
        ml_model=model_with_x,
        conf=ModelConfig(auto_detect_predict_params=True, columns=src_columns),
    )
    ml.load()

    pred_pydantic_model = create_pydantic_predict_model(
        ml, column_validation=False, custom_column_pydantic_model=V1CustomColumnModel
    )

    with pytest.raises(TypeError):
        pred_pydantic_model(X=[{'test_column': 'a'}])
