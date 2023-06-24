from dataclasses import dataclass
from typing import List, Union

from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from mlup.errors import ModelPredictError


class ApiError(BaseModel):
    loc: List
    msg: str
    type: str


class ApiErrorResponse(BaseModel):
    detail: List[ApiError]


@dataclass
class ApiRequestError(Exception):
    message: str
    status_code: int = 400
    type: str = 'request_error'


def api_exception_handler(request: Request, exc: Union[ValidationError, ModelPredictError, ApiRequestError]):
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    if isinstance(exc, ModelPredictError):
        error_res = ApiErrorResponse(detail=[{"loc": [], "msg": str(exc), "type": "predict_error"}])
        http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    elif isinstance(exc, ApiRequestError):
        error_res = ApiErrorResponse(detail=[{"loc": [], "msg": exc.message, "type": exc.type}])
        http_status = exc.status_code
    else:
        error_res = ApiErrorResponse(detail=exc.errors())

    return JSONResponse(
        content=jsonable_encoder(error_res.dict()),
        status_code=http_status,
    )
