from fastapi import APIRouter, HTTPException, status

from app.schemas.training import TrainingRequest, TrainingResponse
from app.services.model_service import ModelService

router = APIRouter()


@router.post("", response_model=TrainingResponse)
def train_stock_model(request: TrainingRequest) -> TrainingResponse:
    try:
        return ModelService().train(request.symbol)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
