import pandas as pd


FEATURE_COLUMNS = [
    "daily_return",
    "ma_5",
    "ma_10",
    "ma_20",
    "volatility_5",
    "momentum_5",
    "volume_change",
]


def build_features(prices: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required price columns: {missing}")

    frame = prices.copy().sort_index()
    frame["daily_return"] = frame["close"].pct_change()
    frame["ma_5"] = frame["close"].rolling(window=5).mean()
    frame["ma_10"] = frame["close"].rolling(window=10).mean()
    frame["ma_20"] = frame["close"].rolling(window=20).mean()
    frame["volatility_5"] = frame["daily_return"].rolling(window=5).std()
    frame["momentum_5"] = frame["close"] / frame["close"].shift(5) - 1
    frame["volume_change"] = frame["volume"].pct_change()

    if include_target:
        frame["next_close"] = frame["close"].shift(-1)
        frame["target"] = (frame["next_close"] > frame["close"]).where(
            frame["next_close"].notna()
        )

    keep_columns = FEATURE_COLUMNS + (["target"] if include_target else [])
    result = frame[keep_columns].replace([float("inf"), float("-inf")], pd.NA).dropna()
    if include_target:
        result["target"] = result["target"].astype(int)
    return result


def latest_feature_row(prices: pd.DataFrame) -> pd.DataFrame:
    features = build_features(prices, include_target=False)
    if features.empty:
        raise ValueError("Not enough historical data to build prediction features.")
    return features[FEATURE_COLUMNS].tail(1)
