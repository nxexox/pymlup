from typing import List, Dict, Tuple

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from mlup.constants import DEFAULT_X_ARG_NAME, IS_X
from mlup.ml.model import MLupModel


_openapi_types_map = {
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
}


def _fix_scheme_by_paths(point: Dict, response_codes: List[str]):
    for http_method_name, http_method_val in point.items():
        # Remove common codes
        for resp_code in response_codes:
            if resp_code in http_method_val["responses"]:
                del http_method_val["responses"][resp_code]
        # Fix method in operationId, if handler have many http methods.
        if not http_method_val['operationId'].endswith(http_method_name.lower()):
            http_method_val['operationId'] = (
                http_method_val['operationId'].rsplit('_', 1)[0] + f'_{http_method_name.lower()}'
            )

    return point


def make_columns_object_openapi_scheme(src_columns: List[Dict]) -> Tuple[Dict, List]:
    cols_openapi_config = {}
    required_columns = []
    for col_config in src_columns:
        col_name, col_type = col_config["name"], col_config.get("type", "str")
        col_required, col_default = col_config.get("required", True), col_config.get("default", None)

        _col_config = {"type": _openapi_types_map[col_type.lower()]}
        title = []
        if col_default is not None:
            title.append("Default")
            _col_config["default"] = col_default
        if col_required:
            required_columns.append(col_name)
        else:
            title.append("Optional")

        title.append(col_type.capitalize())

        if col_config.get(IS_X, False):
            _col_config["items"] = {"$ref": "#/components/schemas/ColumnsForPredict"}
        cols_openapi_config[col_name] = {"title": ' '.join(title), **_col_config}

    return cols_openapi_config, required_columns


def generate_openapi_schema(app: FastAPI, ml: MLupModel):
    """
    openapi_schema["info"] = {
        "title": DOCS_TITLE,
        "version": DOCS_VERSION,
        "description": "Learn about programming language history!",
        "termsOfService": "http://programming-languages.com/terms/",
        "contact": {
            "name": "Get Help with this API",
            "url": "http://www.programming-languages.com/help",
            "email": "support@programming-languages.com"
        },
        "license": {
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
        },
    }
    """
    # TODO: Optimize, dublicate code in api_validation.py
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        openapi_version=app.openapi_version,
        terms_of_service=app.terms_of_service,
        contact=app.contact,
        license_info=app.license_info,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
    )

    if "/health" in openapi_schema["paths"]:
        openapi_schema["paths"]["/health"] = _fix_scheme_by_paths(
            openapi_schema["paths"]["/health"], ['422', '429', '500']
        )
    if "/info" in openapi_schema["paths"]:
        openapi_schema["paths"]["/info"] = _fix_scheme_by_paths(
            openapi_schema["paths"]["/info"], ['422', '429', '500']
        )

    openapi_schema["paths"]["/predict"] = {
      "post": {
        "summary": "Predict",
        "operationId": "predict_predict_post",
        "requestBody": {
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/PredictItems"
              }
            }
          },
          "required": True
        },
        "responses": {
          "200": {
            "description": "Successful Response",
            "content": {
              "application/json": {
                "schema": {}
              }
            }
          },
          '422': {
            'description': 'Error with validation input data',
            'content': {
              'application/json': {
                'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}
              }
            }
          },
          '429': {
            'description': 'Throttling input request error',
            'content': {
              'application/json': {
                'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}
              }
            }
          },
          '500': {
            'description': 'Error with predict process exception',
            'content': {
              'application/json': {
                'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}
              }
            }
          }
        }
      }
    }

    cols_openapi_config, required_columns = {}, []
    if ml.conf.columns:
        cols_openapi_config, required_columns = make_columns_object_openapi_scheme(ml.conf.columns)

    predict_args_config = {
        DEFAULT_X_ARG_NAME: {
            "title": "Data for predict",
            "type": "array",
            "items": {"$ref": "#/components/schemas/ColumnsForPredict"},
        }
    }
    required_args = [DEFAULT_X_ARG_NAME]
    if ml.conf.auto_detect_predict_params:
        predict_args_config, required_args = make_columns_object_openapi_scheme(ml._predict_arguments)

    openapi_schema["components"]["schemas"]["PredictItems"] = {
        "title": "Predict items",
        "required": required_args,
        "type": "object",
        "properties": predict_args_config
    }
    openapi_schema["components"]["schemas"]["ColumnsForPredict"] = {
        "title": "Columns for predict",
        "required": required_columns,
        "type": "object",
        "properties": cols_openapi_config,
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def openapi_schema(app: FastAPI, ml: MLupModel):
    if not app.openapi_schema:
        app.openapi_schema = generate_openapi_schema(app, ml)
    return app.openapi_schema
