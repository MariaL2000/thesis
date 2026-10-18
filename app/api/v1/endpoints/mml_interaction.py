from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models.models import MLModelRun, TranscriptomeReport, User
from app.services.database import get_db
from app.services.mml_service import entrenar_comparar_modelos_ml_background

router = APIRouter(prefix="/ml", tags=["Machine Learning"])


@router.post("/entrenar-interacciones/{user_id}")
async def iniciar_entrenamiento_ml(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    try:
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"El usuario con ID {user_id} no está registrado.")

        transcriptome = (
            db.query(TranscriptomeReport)
            .filter_by(user_id=user_id)
            .order_by(TranscriptomeReport.id.desc())
            .first()
        )
        if not transcriptome:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró transcriptoma para el usuario {user_id}.",
            )

        background_tasks.add_task(
            entrenar_comparar_modelos_ml_background, user_id, transcriptome.id
        )

        return {
            "status": "procesando",
            "user_id": user_id,
            "transcriptome_id": transcriptome.id,
            "message": f"ML iniciado para usuario {user_id} (transcriptoma #{transcriptome.id}).",
            "metrics_get": f"/api/v1/ml/metricas/{user_id}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar ML: {str(e)}")


@router.get("/metricas/{user_id}")
def obtener_metricas_ml(user_id: int, db: Session = Depends(get_db)):
    """GET de métricas / runs de ML cuando el entrenamiento termina."""
    runs = (
        db.query(MLModelRun)
        .filter_by(user_id=user_id)
        .order_by(MLModelRun.trained_at.desc())
        .limit(20)
        .all()
    )
    if not runs:
        return {
            "status": "pendiente",
            "user_id": user_id,
            "mensaje": "Aún no hay corridas de ML para este usuario.",
            "runs": [],
        }

    return {
        "status": "completado",
        "user_id": user_id,
        "runs": [
            {
                "model_version": r.model_version,
                "model_name": r.model_name,
                "transcriptome_id": r.transcriptome_id,
                "training_dataset": r.training_dataset,
                "features_used": r.features_used,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1_score": r.f1_score,
                "roc_auc": r.roc_auc,
                "pr_auc": r.pr_auc,
                "trained_at": r.trained_at,
                "plots_dir": r.plots_dir,
            }
            for r in runs
        ],
    }
