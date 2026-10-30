from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.adapter.http.endpoints.predict import router
from app.adapter.http.middleware import RequestLoggingMiddleware
from app.bootstrap.lifespan import create_lifespan
from app.core.config.settings import settings


def create_app(app_settings=settings):
    app = FastAPI(lifespan=create_lifespan(app_settings))
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def redirect_to_docs():
        return RedirectResponse(url="/docs")

    return app
