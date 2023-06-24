from typing import List, Dict, Tuple

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from mlup.constants import DEFAULT_X_ARG_NAME, IS_X
from mlup.interfaces import MLupModelInterface


_openapi_types_map = {
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "list": "array",
}


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


def generate_openapi_schema(app: FastAPI, mlup_model: MLupModelInterface):
    '''
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
    '''
    # TODO: Optimize, dublicate code in api_validation.py
    openapi_schema = get_openapi(
        title=f"MLup web application with model: {mlup_model.name} v{mlup_model.version}.",
        version=f"v{mlup_model.version}",
        description="Web application for use ml model in web. TODO: Added long and sweat information.",
        openapi_version=app.openapi_version,
        terms_of_service=app.terms_of_service,
        contact=app.contact,
        license_info=app.license_info,
        routes=app.routes,
        tags=app.openapi_tags,
        servers=app.servers,
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
          "422": {
            "description": "Validation Error",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ApiErrorResponse"
                }
              }
            }
          }
        }
      }
    }

    cols_openapi_config, required_columns = {}, []
    if mlup_model.columns:
        cols_openapi_config, required_columns = make_columns_object_openapi_scheme(mlup_model.columns)

    predict_args_config = {
        DEFAULT_X_ARG_NAME: {
            "title": "Data for predict",
            "type": "array",
            "items": {"$ref": "#/components/schemas/ColumnsForPredict"},
        }
    }
    required_args = [DEFAULT_X_ARG_NAME]
    if mlup_model.auto_detect_predict_params:
        predict_args_config, required_args = make_columns_object_openapi_scheme(mlup_model._predict_arguments)

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


def openapi_schema(app: FastAPI, mlup_model: MLupModelInterface):
    if not app.openapi_schema:
        app.openapi_schema = generate_openapi_schema(app, mlup_model)
    return app.openapi_schema
