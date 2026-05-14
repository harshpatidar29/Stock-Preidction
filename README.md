# Stock Predictor API

Educational FastAPI project that predicts whether a stock may move `UP` or `DOWN`
on the next trading day using Alpha Vantage historical prices and a simple
scikit-learn classifier.

This project is for learning only. It is not financial advice and should not be
used to make trading decisions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Alpha Vantage API key to `.env`:

```text
ALPHA_VANTAGE_API_KEY=your_real_key
```

## Run

```bash
python run.py
```

Open the API docs at `http://localhost:8000/docs`.

## Endpoints

### Health

```bash
curl http://localhost:8000/health
```

### Train

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

This fetches recent daily history from Alpha Vantage, builds technical features,
trains a Random Forest classifier, and saves it to
`app/models/stock_direction_model.pkl`.

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL"}'
```

Example response:

```json
{
  "symbol": "AAPL",
  "prediction": "UP",
  "confidence": 0.67
}
```

## Flow

1. `/train` fetches recent historical OHLCV data from Alpha Vantage using the
   free-tier `compact` output size.
2. The feature service creates daily return, moving averages, volatility,
   momentum, and volume change.
3. The target is `1` when the next close is higher than the current close, and
   `0` otherwise.
4. The model is saved locally with `joblib`.
5. `/predict` fetches recent data, creates the latest feature row, loads the
   saved model, and returns `UP` or `DOWN` with confidence.

## Tests

```bash
pytest
```
