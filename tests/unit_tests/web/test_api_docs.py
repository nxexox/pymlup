import copy
from typing import Optional
from unittest import TestCase

from pydantic import BaseModel as PydanticBaseModel

from mlup.constants import IS_X, DEFAULT_X_ARG_NAME, WebAppArchitecture
from mlup.up import UP, Config
from mlup.web.api_docs import make_columns_object_openapi_scheme, generate_openapi_schema


assertDictEqual = TestCase().assertDictEqual


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


openapi_cols = {
    'Float': {'title': 'Float', 'type': 'number'},
    'FloatDefault': {'title': 'Default Float', 'type': 'number', 'default': 1.4},
    'FloatRequired': {'title': 'Float', 'type': 'number'},
    'FloatNotRequired': {'title': 'Optional Float', 'type': 'number'},
    'FloatNotRequiredDefault': {'title': 'Default Optional Float', 'type': 'number', 'default': 1.4},
    'FloatRequiredDefault': {'title': 'Default Float', 'type': 'number', 'default': 1.4},
    'Int': {'title': 'Int', 'type': 'integer'},
    'IntDefault': {'title': 'Default Int', 'type': 'integer', 'default': 4},
    'IntRequired': {'title': 'Int', 'type': 'integer'},
    'IntNotRequired': {'title': 'Optional Int', 'type': 'integer'},
    'IntNotRequiredDefault': {'title': 'Default Optional Int', 'type': 'integer', 'default': 4},
    'IntRequiredDefault': {'title': 'Default Int', 'type': 'integer', 'default': 4},
    'Str': {'title': 'Str', 'type': 'string'},
    'StrDefault': {'title': 'Default Str', 'type': 'string', 'default': 'str'},
    'StrRequired': {'title': 'Str', 'type': 'string'},
    'StrNotRequired': {'title': 'Optional Str', 'type': 'string'},
    'StrNotRequiredDefault': {'title': 'Default Optional Str', 'type': 'string', 'default': 'str'},
    'StrRequiredDefault': {'title': 'Default Str', 'type': 'string', 'default': 'str'},
    'Bool': {'title': 'Bool', 'type': 'boolean'},
    'BoolDefault': {'title': 'Default Bool', 'type': 'boolean', 'default': True},
    'BoolRequired': {'title': 'Bool', 'type': 'boolean'},
    'BoolNotRequired': {'title': 'Optional Bool', 'type': 'boolean'},
    'BoolNotRequiredDefault': {'title': 'Default Optional Bool', 'type': 'boolean', 'default': True},
    'BoolRequiredDefault': {'title': 'Default Bool', 'type': 'boolean', 'default': True}
}
openapi_required_cols = [
    'Float', 'FloatDefault', 'FloatRequired', 'FloatRequiredDefault',
    'Int', 'IntDefault', 'IntRequired', 'IntRequiredDefault',
    'Str', 'StrDefault', 'StrRequired', 'StrRequiredDefault',
    'Bool', 'BoolDefault', 'BoolRequired', 'BoolRequiredDefault'
]

openapi_full_scheme = {
    'openapi': '3.0.2',
    'info': {
        'title': 'MLup web application with model: MyFirstMLupModel v1.0.0.0.',
        'description': 'Web application for use MyFirstMLupModel v1.0.0.0 in web.',
        'version': '1.0.0.0'
    },
    'paths': {
        '/health': {
            'get': {
                'summary': 'Http Health',
                'operationId': 'http_health_health_get',
                'responses': {
                    '200': {
                        'description': 'Successful Response',
                        'content': {
                            'application/json': {
                                'schema': {}
                            }
                        }
                    }
                }
            },
            'options': {
                'summary': 'Http Health',
                'operationId': 'http_health_health_options',
                'responses': {
                    '200': {
                        'description': 'Successful Response',
                        'content': {
                            'application/json': {
                                'schema': {}
                            }
                        }
                    }
                }
            },
            'head': {
                'summary': 'Http Health',
                'operationId': 'http_health_health_head',
                'responses': {
                    '200': {
                        'description': 'Successful Response',
                        'content': {
                            'application/json': {
                                'schema': {}
                            }
                        }
                    },
                }
            },
        },
        '/info': {
            'get': {
                'summary': 'Info',
                'operationId': 'info_info_get',
                'responses': {
                    '200': {
                        'description': 'Successful Response',
                        'content': {
                            'application/json': {
                                'schema': {}
                            }
                        }
                    }
                }
            }
        },
        '/predict': {
            'post': {
                'summary': 'Predict',
                'operationId': 'predict_predict_post',
                'requestBody': {
                    'content': {
                        'application/json': {
                            'schema': {
                                '$ref': '#/components/schemas/PredictItems'
                            }
                        }
                    },
                    'required': True
                },
                'responses': {
                    '200': {
                        'description': 'Successful Response',
                        'content': {
                            'application/json': {
                                'schema': {}
                            }
                        }
                    },
                    '422': {
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref': '#/components/schemas/ApiErrorResponse'
                                }
                            }
                        },
                        'description': 'Error with validation input data'
                    },
                    '429': {
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref': '#/components/schemas/ApiErrorResponse'
                                }
                            }
                        },
                        'description': 'Throttling input request error'
                    },
                    '500': {
                        'content': {
                            'application/json': {
                                'schema': {
                                    '$ref': '#/components/schemas/ApiErrorResponse'
                                }
                            }
                        },
                        'description': 'Error with predict process exception'
                    }
                }
            }
        }
    },
    'components': {
        'schemas': {
            'ApiError': {
                'title': 'ApiError',
                'required': ['loc', 'msg', 'type'],
                'type': 'object',
                'properties': {
                    'loc': {
                        'title': 'Loc',
                        'type': 'array',
                        'items': {}
                    },
                    'msg': {
                        'title': 'Msg',
                        'type': 'string'
                    },
                    'type': {
                        'title': 'Type',
                        'type': 'string'
                    }
                }
            },
            'ApiErrorResponse': {
                'title': 'ApiErrorResponse',
                'required': ['detail'],
                'type': 'object',
                'properties': {
                    'detail': {
                        'title': 'Detail',
                        'type': 'array',
                        'items': {
                            '$ref': '#/components/schemas/ApiError'
                        }
                    },
                    'predict_id': {
                        'title': 'Predict Id',
                        'type': 'string'
                    }
                },
            },
            'PredictItems': {
                'title': 'Predict items',
                'required': ['X'],
                'type': 'object',
                'properties': {
                    'X': {
                        'title': 'List',
                        'type': 'array',
                        'items': {
                            '$ref': '#/components/schemas/ColumnsForPredict'
                        }
                    },
                    'test_param': {
                        'title': 'Default Optional Bool',
                        'type': 'boolean',
                        'default': False
                    }
                }
            },
            'ColumnsForPredict': {
                'title': 'Columns for predict',
                'required': [],
                'type': 'object',
                'properties': {}
            }
        }
    }
}
openapi_is_long_predict_method = {
    '/get-predict/{predict_id}': {
        'get': {
            'summary': 'Get Predict Result',
            'operationId': 'get_predict_result_get_predict__predict_id__get',
            'parameters': [
                {
                    'required': True,
                    'schema': {'title': 'Predict Id', 'type': 'string'},
                    'name': 'predict_id',
                    'in': 'path'
                }
            ],
            'responses': {
                '200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}},
                '422': {
                    'description': 'Error with validation input data',
                    'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}}}
                },
                '429': {
                    'description': 'Throttling input request error',
                    'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}}}
                },
                '500': {
                    'description': 'Error with predict process exception',
                    'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}}}
                }
            }
        }
    }
}


def test_make_columns_object_openapi_scheme():
    pred_cols, pred_required_cols = make_columns_object_openapi_scheme(src_columns)

    assertDictEqual(pred_cols, openapi_cols)
    assert pred_required_cols == openapi_required_cols


def test_make_columns_object_openapi_scheme_with_IS_X():
    test_columns = copy.deepcopy(src_columns)
    test_columns[0][IS_X] = True
    pred_cols, pred_required_cols = make_columns_object_openapi_scheme(test_columns)

    test_openapi_cols = copy.deepcopy(openapi_cols)
    test_openapi_cols['Float']['items'] = {"$ref": "#/components/schemas/ColumnsForPredict"}
    assertDictEqual(pred_cols, test_openapi_cols)
    assert pred_required_cols == openapi_required_cols


def test_generate_openapi_schema(print_model):
    up = UP(ml_model=print_model)
    # Load ml with web
    up.ml.load()
    up.web.load()
    generated_scheme = generate_openapi_schema(up.web.app, up.ml)
    openapi_full_scheme['openapi'] = generated_scheme['openapi']
    assertDictEqual(generated_scheme, openapi_full_scheme)


def test_generate_openapi_scheme_with_default_X_and_custom_columns(print_model):
    up = UP(
        ml_model=print_model,
        conf=Config(
            auto_detect_predict_params=False,
            columns=src_columns,
        )
    )
    # Load ml with web
    up.ml.load()
    up.web.load()
    generated_scheme = generate_openapi_schema(up.web.app, up.ml)

    _openapi_full_scheme = copy.deepcopy(openapi_full_scheme)
    _scheme_predict_items = _openapi_full_scheme['components']['schemas']['PredictItems']
    _scheme_predict_items['required'] = [DEFAULT_X_ARG_NAME]
    _scheme_predict_items['properties'] = {
        DEFAULT_X_ARG_NAME: {
            'title': 'Data for predict',
            'type': 'array',
            'items': {
                '$ref': '#/components/schemas/ColumnsForPredict'
            }
        }
    }
    _scheme_columns_for_predict = _openapi_full_scheme['components']['schemas']['ColumnsForPredict']
    _scheme_columns_for_predict['required'] = openapi_required_cols
    _scheme_columns_for_predict['properties'] = openapi_cols

    _openapi_full_scheme['openapi'] = generated_scheme['openapi']
    assertDictEqual(generated_scheme, _openapi_full_scheme)


def test_generate_openapi_scheme_with_default_X_without_columns(print_model):
    up = UP(
        ml_model=print_model,
        conf=Config(
            auto_detect_predict_params=False,
        )
    )
    # Load ml with web
    up.ml.load()
    up.web.load()
    generated_scheme = generate_openapi_schema(up.web.app, up.ml)

    _openapi_full_scheme = copy.deepcopy(openapi_full_scheme)
    _scheme_predict_items = _openapi_full_scheme['components']['schemas']['PredictItems']
    _scheme_predict_items['required'] = [DEFAULT_X_ARG_NAME]
    _scheme_predict_items['properties'] = {
        DEFAULT_X_ARG_NAME: {
            'title': 'Data for predict',
            'type': 'array',
            'items': {
                '$ref': '#/components/schemas/ColumnsForPredict'
            }
        }
    }

    openapi_full_scheme['openapi'] = generated_scheme['openapi']
    assertDictEqual(generated_scheme, _openapi_full_scheme)


def test_generate_openapi_scheme_with_auto_analyze_X_with_custom_columns(print_model):
    up = UP(
        ml_model=print_model,
        conf=Config(
            auto_detect_predict_params=True,
            columns=src_columns,
        )
    )
    # Load ml with web
    up.ml.load()
    up.web.load()
    generated_scheme = generate_openapi_schema(up.web.app, up.ml)

    _openapi_full_scheme = copy.deepcopy(openapi_full_scheme)
    _scheme_predict_items = _openapi_full_scheme['components']['schemas']['PredictItems']
    _scheme_predict_items['required'] = ['X']
    _scheme_predict_items['properties'] = {
        'X': {'title': 'List', 'type': 'array', 'items': {'$ref': '#/components/schemas/ColumnsForPredict'}},
        'test_param': {'title': 'Default Optional Bool', 'type': 'boolean', 'default': False},
    }
    _scheme_columns_for_predict = _openapi_full_scheme['components']['schemas']['ColumnsForPredict']
    _scheme_columns_for_predict['required'] = openapi_required_cols
    _scheme_columns_for_predict['properties'] = openapi_cols

    _openapi_full_scheme['openapi'] = generated_scheme['openapi']
    assertDictEqual(generated_scheme, _openapi_full_scheme)


def test_generate_openapi_scheme_with_auto_analyze_X_without_columns(print_model):
    up = UP(
        ml_model=print_model,
        conf=Config(auto_detect_predict_params=True)
    )
    # Load ml with web
    up.ml.load()
    up.web.load()
    generated_scheme = generate_openapi_schema(up.web.app, up.ml)

    openapi_full_scheme['openapi'] = generated_scheme['openapi']
    assertDictEqual(generated_scheme, openapi_full_scheme)


def test_generate_openapi_scheme_with_is_long_predict(print_model):
    up = UP(
        ml_model=print_model,
        conf=Config(auto_detect_predict_params=False)
    )
    # Load ml with web
    up.ml.load()
    up.web.conf.mode = WebAppArchitecture.worker_and_queue
    up.web.conf.is_long_predict = True
    up.web.load()
    generated_scheme = generate_openapi_schema(up.web.app, up.ml)

    _openapi_full_scheme = copy.deepcopy(openapi_full_scheme)
    _openapi_full_scheme['paths'].update(openapi_is_long_predict_method)
    _scheme_predict_items = _openapi_full_scheme['components']['schemas']['PredictItems']
    _scheme_predict_items['required'] = [DEFAULT_X_ARG_NAME]
    _scheme_predict_items['properties'] = {
        DEFAULT_X_ARG_NAME: {
            'title': 'Data for predict',
            'type': 'array',
            'items': {
                '$ref': '#/components/schemas/ColumnsForPredict'
            }
        }
    }

    _openapi_full_scheme['openapi'] = generated_scheme['openapi']
    assertDictEqual(generated_scheme, _openapi_full_scheme)


def test_generate_openapi_scheme_with_add_custom_handler(print_model):
    class CustomRequestData(PydanticBaseModel):
        i: int
        f: Optional[float] = None
        s: str = 'test string'
        b: Optional[bool] = False

    async def test_api_handler(item_id: int, data: CustomRequestData):
        return data.dict()

    up = UP(
        ml_model=print_model,
        conf=Config(auto_detect_predict_params=False)
    )
    # Load ml with web
    up.ml.load()
    # Add custom handler
    up.web.load()
    up.web.app.add_api_route("/test-api-handler/{item_id}", test_api_handler, methods=["POST"])
    generated_scheme = generate_openapi_schema(up.web.app, up.ml)

    _openapi_full_scheme = copy.deepcopy(openapi_full_scheme)
    _scheme_predict_items = _openapi_full_scheme['components']['schemas']['PredictItems']
    _scheme_predict_items['required'] = [DEFAULT_X_ARG_NAME]
    _scheme_predict_items['properties'] = {
        DEFAULT_X_ARG_NAME: {
            'title': 'Data for predict',
            'type': 'array',
            'items': {
                '$ref': '#/components/schemas/ColumnsForPredict'
            }
        }
    }
    _openapi_full_scheme['paths']['/test-api-handler/{item_id}'] = {
        'post': {
            'summary': 'Test Api Handler',
            'operationId': 'test_api_handler_test_api_handler__item_id__post',
            'parameters': [
                {
                    'required': True,
                    'schema': {'title': 'Item Id', 'type': 'integer'},
                    'name': 'item_id',
                    'in': 'path'
                }
            ],
            'requestBody': {
                'content': {'application/json': {'schema': {'$ref': '#/components/schemas/CustomRequestData'}}},
                'required': True
            },
            'responses': {
                '200': {'description': 'Successful Response', 'content': {'application/json': {'schema': {}}}},
                '422': {
                    'description': 'Error with validation input data',
                    'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}}}
                },
                '429': {
                    'description': 'Throttling input request error',
                    'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}}}
                },
                '500': {
                    'description': 'Error with predict process exception',
                    'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiErrorResponse'}}}
                }
            }
        }
    }
    _openapi_full_scheme['components']['schemas']['CustomRequestData'] = {
        'title': 'CustomRequestData',
        'required': ['i'],
        'type': 'object',
        'properties': {
            'i': {'title': 'I', 'type': 'integer'},
            'f': {'title': 'F', 'type': 'number'},
            's': {'title': 'S', 'type': 'string', 'default': 'test string'},
            'b': {'title': 'B', 'type': 'boolean', 'default': False}
        }
    }

    _openapi_full_scheme['openapi'] = generated_scheme['openapi']
    assertDictEqual(generated_scheme, _openapi_full_scheme)
