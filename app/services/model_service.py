from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.schemas.training import TrainingResponse
from app.services.alpha_vantage import AlphaVantageClient
from app.services.features import FEATURE_COLUMNS, build_features


class ModelService:
    def __init__(
        self,
        data_client: AlphaVantageClient | None = None,
        model_path: Path | None = None,
    ) -> None:
        self.data_client = data_client or AlphaVantageClient()
        self.model_path = model_path or settings.model_path

    def train(self, symbol: str) -> TrainingResponse:
        prices = self.data_client.get_daily_prices(symbol, outputsize="compact")
        dataset = build_features(prices, include_target=True)
        if len(dataset) < 40:
            raise ValueError("Not enough historical data to train the model.")

        x = dataset[FEATURE_COLUMNS]
        y = dataset["target"]
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            shuffle=False,
        )

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)

        return TrainingResponse(
            symbol=symbol,
            rows_used=len(dataset),
            accuracy=round(float(accuracy), 4),
            model_path=str(self.model_path),
        )
