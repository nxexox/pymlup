from dataclasses import dataclass
from typing import List, Union, Optional

from fastapi import Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from mlup.constants import PREDICT_ID_HEADER
from mlup.errors import PredictError, PredictWaitResultError, PredictTransformDataError, \
    PredictValidationInnerDataError


class ApiError(BaseModel):
    loc: List
    msg: str
    type: str


class ApiErrorResponse(BaseModel):
    detail: List[ApiError]
    predict_id: Optional[str] = None


@dataclass
class ApiRequestError(Exception):
    message: str
    status_code: int = 400
    type: str = 'request_error'


def api_exception_handler(request: Request, exc: Union[ValidationError, ApiRequestError]):
    headers = {}
    error_res = ApiErrorResponse(detail=exc.errors())
    if hasattr(exc, 'predict_id'):
        error_res.predict_id = exc.predict_id
        headers[PREDICT_ID_HEADER] = exc.predict_id
    return JSONResponse(
        content=jsonable_encoder(error_res.dict()),
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        headers=headers,
    )


def api_request_error_handler(request: Request, exc: ApiRequestError):
    headers = {}
    error_res = ApiErrorResponse(detail=[{"loc": [], "msg": exc.message, "type": exc.type}])
    if hasattr(exc, 'predict_id'):
        error_res.predict_id = exc.predict_id
        headers[PREDICT_ID_HEADER] = exc.predict_id
    return JSONResponse(
        content=jsonable_encoder(error_res.dict()),
        status_code=exc.status_code,
        headers=headers,
    )


def predict_errors_handler(
    request: Request,
    exc: Union[PredictError, PredictWaitResultError, PredictTransformDataError, PredictValidationInnerDataError],
):
    headers = {}
    error_res = ApiErrorResponse(detail=[{"loc": [], "msg": str(exc), "type": exc.type}])
    if hasattr(exc, 'predict_id'):
        error_res.predict_id = exc.predict_id
        headers[PREDICT_ID_HEADER] = exc.predict_id
    return JSONResponse(
        content=jsonable_encoder(error_res.dict()),
        status_code=exc.http_status,
        headers=headers,
    )
