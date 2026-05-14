from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.services.prediction_service import PredictionService


class FakeClient:
    def get_daily_prices(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=45, freq="D")
        close = [100 + index for index in range(45)]
        return pd.DataFrame(
            {
                "open": close,
                "high": [value + 1 for value in close],
                "low": [value - 1 for value in close],
                "close": close,
                "volume": [1000 + index * 10 for index in range(45)],
            },
            index=dates,
        )


def test_prediction_fails_clearly_when_model_missing(tmp_path: Path) -> None:
    service = PredictionService(data_client=FakeClient(), model_path=tmp_path / "missing.pkl")

    with pytest.raises(FileNotFoundError, match="No trained model found"):
        service.predict("AAPL")


def test_prediction_response_shape_with_saved_model(tmp_path: Path) -> None:
    x = pd.DataFrame(
        {
            "daily_return": [0.01, -0.01, 0.02, -0.02],
            "ma_5": [101, 102, 103, 104],
            "ma_10": [100, 101, 102, 103],
            "ma_20": [99, 100, 101, 102],
            "volatility_5": [0.01, 0.02, 0.01, 0.02],
            "momentum_5": [0.03, -0.03, 0.04, -0.04],
            "volume_change": [0.01, 0.02, 0.01, 0.02],
        }
    )
    y = [1, 0, 1, 0]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(x, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump(model, model_path)

    service = PredictionService(data_client=FakeClient(), model_path=model_path)
    response = service.predict("AAPL")

    assert response.symbol == "AAPL"
    assert response.prediction in {"UP", "DOWN"}
    assert 0 <= response.confidence <= 1
