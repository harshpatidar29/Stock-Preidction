import pandas as pd

from app.services.features import FEATURE_COLUMNS, build_features, latest_feature_row


def sample_prices(rows: int = 35) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = [100 + index for index in range(rows)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "volume": [1000 + index * 10 for index in range(rows)],
        },
        index=dates,
    )


def test_build_features_creates_expected_columns_and_target() -> None:
    features = build_features(sample_prices(), include_target=True)

    assert list(features.columns) == FEATURE_COLUMNS + ["target"]
    assert not features.isna().any().any()
    assert set(features["target"].unique()) == {1}


def test_latest_feature_row_returns_one_row() -> None:
    latest = latest_feature_row(sample_prices())

    assert list(latest.columns) == FEATURE_COLUMNS
    assert len(latest) == 1
