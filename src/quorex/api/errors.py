from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

class ApiError(Exception):
    status: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

class InvalidApiKey(ApiError):
    status, code = 401, "invalid_api_key"

class UserNotFound(ApiError):
    status, code = 404, "user_not_found"

class FactNotFound(ApiError):
    status, code = 404, "fact_not_found"

class InvalidRequest(ApiError):
    status, code = 422, "invalid_request"

class InvalidTimeRange(ApiError):
    status, code = 422, "invalid_time_range"

class ConcurrentWriteError(ApiError):
    status, code = 409, "concurrent_write"

class FactRejected(ApiError):
    status, code = 409, "fact_rejected"

class LLMUnvailable(ApiError):
    status, code = 503, "llm_unavailable"

def _payload(request: Request, code: str, message: str, details: Any | None) -> dict:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": getattr(request.state, "request_id", None)}

    if details is not None:
        body["details"] = details
    return {"error": body}

def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content=_payload(request, exc.code, exc.message, exc.details))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content=_payload(request, "invalid_request", "requête invalide", details))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Jamais str(e) ni de stack dans le corps (audi legacy, 04-qualite.
        return JSONResponse(status_code=500, content=_payload(request, "internal_error", "erreur interne", None))