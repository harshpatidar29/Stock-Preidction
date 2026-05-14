from pydantic import BaseModel, Field, field_validator


class TrainingRequest(BaseModel):
    symbol: str = Field(..., min_length=1, examples=["AAPL"])

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()


class TrainingResponse(BaseModel):
    symbol: str
    rows_used: int
    accuracy: float = Field(..., ge=0, le=1)
    model_path: str
