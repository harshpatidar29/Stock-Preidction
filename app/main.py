from fastapi import FastAPI

from app.api.routes import health, predict, train
from app.core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Educational API for predicting next-day stock direction.",
    )

    app.include_router(health.router)
    app.include_router(train.router, prefix="/train", tags=["training"])
    app.include_router(predict.router, prefix="/predict", tags=["prediction"])

    return app


app = create_app()
