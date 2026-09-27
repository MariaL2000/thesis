from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.services.database import get_db
from app.models.models import User, TranscriptomeReport
from app.services.mml_service import entrenar_comparar_modelos_ml_background

router = APIRouter(prefix="/ml", tags=["Machine Learning"])

@router.post("/entrenar-interacciones/{user_id}")
async def iniciar_entrenamiento_ml(
    user_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Inicia el entrenamiento de Machine Learning (Logistic Regression, Random Forest y XGBoost)
    en segundo plano de forma automática para el transcriptoma más reciente del usuario,
    almacenando todas las métricas de evaluación en la base de datos.
    """
    try:
        # 1. Validar que el usuario exista
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"El usuario con ID {user_id} no está registrado.")

        # 2. Buscar automáticamente el transcriptoma más reciente del usuario
        transcriptome = (
            db.query(TranscriptomeReport)
            .filter_by(user_id=user_id)
            .order_by(TranscriptomeReport.id.desc())
            .first()
        )
        
        if not transcriptome:
            raise HTTPException(
                status_code=404, 
                detail=f"No se encontró ningún transcriptoma registrado para el usuario {user_id}. Sube un archivo primero."
            )

        transcriptome_id = transcriptome.id

        # 3. Encolar la tarea pesada de Machine Learning en segundo plano con el ID detectado
        background_tasks.add_task(entrenar_comparar_modelos_ml_background, user_id, transcriptome_id)

        return {
            "status": "procesando",
            "user_id": user_id,
            "transcriptome_id": transcriptome_id,
            "message": f"Pipeline de Machine Learning iniciado en segundo plano para el usuario {user_id} (Transcriptoma #{transcriptome_id})."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar el entrenamiento de ML: {str(e)}")