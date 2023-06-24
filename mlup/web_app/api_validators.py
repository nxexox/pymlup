import logging
from typing import List, Any, Dict, Tuple, Optional, Type

from pydantic import BaseModel, validator, create_model, Field

from mlup.constants import IS_X, DEFAULT_X_ARG_NAME
from mlup.interfaces import MLupModelInterface


logger = logging.getLogger('MLup')


class MlModelPredictRequest(BaseModel):
    pass


def validator_by_required(v):
    if v is None:
        raise ValueError('Value is required DEIS')
    return v


def validator_by_int(v):
    if not isinstance(v, int):
        raise ValueError('Value not int type')
    return v


def validator_by_float(v):
    if not isinstance(v, float):
        raise ValueError('Value not float type')
    return v


def validator_by_bool(v):
    if not isinstance(v, bool):
        raise ValueError('Value not bool type')
    return v


def validator_by_str(v):
    if not isinstance(v, str):
        raise ValueError('Value not string type')
    return v


_type_validators = {
    'int': validator_by_int,
    'float': validator_by_float,
    'str': validator_by_str,
    'bool': validator_by_bool,
}


def make_map_pydantic_columns(
    src_columns: List[Dict[str, Any]],
    x_model: Optional[BaseModel] = Any,
) -> Tuple[Dict, Dict]:
    """
    Create Map columns for create pydantic model, by columns from config.
    Use for creating pydantic models in WebApp.

    :param List[Dict[str, Any]] src_columns: List columns from config.
    :param Optional[BaseModel] x_model: Col type from predict method in model.
        Use for auto detect X in model predict method.

    :return: Two dicts.
        The first dict is columns map for func pydantic.create_model.
        The second dict is columns validators for arg __validators__ in pydantic.create_model.
    :rtype: Tuple[Dict, Dict]

    """
    column_types = {
        'int': int,
        'float': float,
        'bool': bool,
        'str': str,
        'list': list,
    }
    __validators__ = {}
    columns_pydantic_format = {}

    for col_config in src_columns:
        # Get col type and create col type validator.
        if col_config.get(IS_X, False):
            col_type = List[x_model]
        else:
            try:
                col_type = column_types[col_config['type'].lower()]
                __validators__[f'{col_config["name"]}_type_validator'] = validator(
                    col_config['name'], allow_reuse=True
                )(_type_validators[col_config['type'].lower()])
            except KeyError:
                logger.warning(
                    f'Field "{col_config["name"]}" has not a type. Please write type for field. '
                    f'Supported types {", ".join(column_types.keys())}.'
                )
                col_type = Any

        field_info = Field(col_config.get('default', ...), title=col_config['name'])

        # Make required validator
        col_data = (col_type, field_info)
        if not col_config.get("required", True):
            __validators__[f'{col_config["name"]}_required_validator'] = validator(
                col_config['name'], allow_reuse=True
            )(validator_by_required)

        columns_pydantic_format[col_config["name"]] = col_data

    return columns_pydantic_format, __validators__


def create_pydantic_predict_model(
    mlup_model: MLupModelInterface,
    column_validation: bool,
    custom_column_pydantic_model: Optional[BaseModel] = None,
) -> Type[MlModelPredictRequest]:
    # If model columns not validate, set Any
    mlup_columns_model = Any

    # If set custom column validation
    if custom_column_pydantic_model is not None:
        mlup_columns_model = custom_column_pydantic_model
    # If MLup column validation IS on
    elif column_validation:
        columns_pydantic_format, __validators__ = make_map_pydantic_columns(mlup_model.columns)
        mlup_columns_model = create_model(
            "MLupModelColumns",
            __validators__=__validators__,
            **columns_pydantic_format,
        )

    # Make first arguments level
    if mlup_model.auto_detect_predict_params:
        pred_args_pydantic_format, __pred_args_validators__ = make_map_pydantic_columns(
            mlup_model._predict_arguments,
            x_model=mlup_columns_model,
        )
        mlup_predict_pydantic_model = create_model(
            "MLupRequestPredictPydanticModel",
            __base__=MlModelPredictRequest,
            __validators__=__pred_args_validators__,
            **pred_args_pydantic_format,
        )
    else:
        mlup_predict_pydantic_model = create_model(
            "MLupRequestPredictPydanticModel",
            __base__=MlModelPredictRequest,
            **{DEFAULT_X_ARG_NAME: (List[mlup_columns_model], list())}
        )
    return mlup_predict_pydantic_model
