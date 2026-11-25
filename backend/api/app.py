import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes import experiments, health, models_api, optimization_api, results
from api.services.experiment_service import ExperimentService
from api.websocket.experiments import router as websocket_router

logger = logging.getLogger(__name__)
service = ExperimentService()


def create_app(experiment_service: ExperimentService | None = None) -> FastAPI:
    @asynccontextmanager
    async def app_lifespan(application: FastAPI):
        active_service = application.state.experiment_service
        active_service.start()
        yield
        active_service.shutdown()

    application = FastAPI(
        title="KV Cache Optimization Lab API",
        version="1.0.0",
        lifespan=app_lifespan,
    )
    origins = [
        origin.strip()
        for origin in os.environ.get(
            "FRONTEND_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        logger.debug(
            "Request validation failed for %s %s",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=422,
            content={
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed.",
                "details": [
                    {
                        "location": [str(part) for part in item["loc"]],
                        "message": item["msg"],
                        "type": item["type"],
                    }
                    for item in error.errors()
                ],
            },
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        _request: Request, error: StarletteHTTPException
    ) -> JSONResponse:
        logger.debug("HTTP %s at %s", error.status_code, _request.url.path)
        if isinstance(error.detail, dict):
            body = error.detail
        else:
            body = {
                "code": "HTTP_ERROR",
                "message": str(error.detail),
                "details": [],
            }
        return JSONResponse(
            status_code=error.status_code,
            content=body,
            headers=error.headers,
        )

    @application.exception_handler(Exception)
    async def internal_error_handler(
        request: Request, error: Exception
    ) -> JSONResponse:
        logger.error(
            "Unhandled API error at %s %s: %s",
            request.method,
            request.url.path,
            error,
            exc_info=error,
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_ERROR",
                "message": "An internal server error occurred.",
                "details": [],
            },
        )

    application.include_router(health.router, prefix="/api/v1")
    application.include_router(models_api.router, prefix="/api/v1")
    application.include_router(optimization_api.router, prefix="/api/v1")
    application.include_router(experiments.router, prefix="/api/v1")
    application.include_router(results.router, prefix="/api/v1")
    application.include_router(websocket_router)
    application.state.experiment_service = experiment_service or service
    return application


app = create_app()
