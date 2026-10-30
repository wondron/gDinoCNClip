"""Service entry point, matching Image-Embedding/backend/main.py."""

from app.bootstrap.application import create_app
from app.core.config.settings import settings
from app.core.observability.logging import setup_logger

app = create_app()


def run():
    import uvicorn

    from app.core.config.constant import BASE_DIR
    setup_logger()
    uvicorn.run("main:app", host=settings.application.host, port=settings.application.port,
                reload=settings.application.reload, log_config=None, app_dir=str(BASE_DIR))


if __name__ == "__main__":
    run()
