from pathlib import Path

import joblib

from app.core.config import settings
from app.schemas.prediction import PredictionResponse
from app.services.alpha_vantage import AlphaVantageClient
from app.services.features import latest_feature_row


class PredictionService:
    def __init__(
        self,
        data_client: AlphaVantageClient | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.data_client = data_client or AlphaVantageClient()
        self.model_path = model_path or settings.model_path

    def predict(self, symbol: str) -> PredictionResponse:
        if not self.model_path.exists():
            raise FileNotFoundError("No trained model found. Train a model with POST /train first.")

        prices = self.data_client.get_daily_prices(symbol, outputsize="compact")
        features = latest_feature_row(prices)
        model = joblib.load(self.model_path)

        prediction_value = int(model.predict(features)[0])
        confidence = self._prediction_confidence(model, features, prediction_value)
        prediction = "UP" if prediction_value == 1 else "DOWN"

        return PredictionResponse(
            symbol=symbol,
            prediction=prediction,
            confidence=round(confidence, 4),
        )

    def _prediction_confidence(self, model: object, features: object, prediction_value: int) -> float:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)
            class_index = classes.index(prediction_value)
            return float(probabilities[class_index])
        return 1.0
