from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Stock Predictor API"
    app_version: str = "0.1.0"
    alpha_vantage_api_key: str = "H02AJYRSZ6ZDIJ2O"
    alpha_vantage_base_url: str = "https://www.alphavantage.co/query"
    model_path: Path = Path("app/models/stock_direction_model.pkl")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
