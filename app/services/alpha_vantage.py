from typing import Any

import pandas as pd
import requests

from app.core.config import settings


class AlphaVantageClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else settings.alpha_vantage_api_key
        self.base_url = base_url or settings.alpha_vantage_base_url

    def get_daily_prices(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is missing. Add it to your .env file.")

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": outputsize,
        }
        response = requests.get(self.base_url, params=params, timeout=20)
        response.raise_for_status()
        payload = response.json()
        return self._parse_daily_response(payload)

    def _parse_daily_response(self, payload: dict[str, Any]) -> pd.DataFrame:
        if "Error Message" in payload:
            raise ValueError(payload["Error Message"])
        if "Note" in payload:
            raise ValueError("Alpha Vantage rate limit reached. Try again later.")
        if "Information" in payload:
            raise ValueError(payload["Information"])

        time_series = payload.get("Time Series (Daily)")
        if not time_series:
            raise ValueError("Alpha Vantage response did not include daily time series data.")

        frame = pd.DataFrame.from_dict(time_series, orient="index")
        frame.index = pd.to_datetime(frame.index)
        frame = frame.rename(
            columns={
                "1. open": "open",
                "2. high": "high",
                "3. low": "low",
                "4. close": "close",
                "5. volume": "volume",
            }
        )
        frame = frame[["open", "high", "low", "close", "volume"]].astype(float)
        frame = frame.sort_index()
        frame.index.name = "date"
        return frame
