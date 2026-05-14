from fastapi import APIRouter, HTTPException, status

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.post("", response_model=PredictionResponse)
def predict_stock_direction(request: PredictionRequest) -> PredictionResponse:
    try:
        return PredictionService().predict(request.symbol)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
