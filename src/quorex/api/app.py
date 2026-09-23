from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import Response

from quorex.api.errors import install_error_handlers
from quorex.api.routes import router
from quorex.contradiction import Resolver
from quorex.ids import new_id
from quorex.normalization import Normalizer
from quorex.settings import Settings, get_settings
from quorex.storage import Database, PostgresFactStore, PostgresTenantStore, load_synonyms

log = structlog.get_logger()


def create_app(settings: Settings | None = None, db: Database | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database = db or Database(settings.database_url)
        facts = PostgresFactStore(database)
        app.state.db = database
        app.state.facts = facts
        app.state.tenants = PostgresTenantStore(database)
        app.state.normalizer = Normalizer(load_synonyms(database))
        app.state.resolver = Resolver(facts)
        log.info("quorex.start", llm_provider=settings.llm_provider,
                 embedding_model=settings.embedding_model, synonyms=len(app.state.normalizer))
        yield
        if db is None:
            database.dispose()

    app = FastAPI(title="QUOREX", version="0.1.0", lifespan=lifespan, docs_url="/docs", redoc_url=None)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next) -> Response:
        request.state.request_id = str(new_id())
        response = await call_next(request)
        response.headers["X-Request-Id"] = request.state.request_id
        return response

    install_error_handlers(app)
    app.include_router(router)
    return app