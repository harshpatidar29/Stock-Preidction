from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=1, examples=["AAPL"])

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()


class PredictionResponse(BaseModel):
    symbol: str
    prediction: str
    confidence: float = Field(..., ge=0, le=1)
