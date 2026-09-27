from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import Optional
from sqlalchemy.orm import Session
from app.services.database import get_db
from app.models.models import User, TranscriptomeReport
from app.services.validation import entrenar_comparar_modelos_ml_background

router = APIRouter()

@router.post("/ml/pipeline-completo/{user_id}")
async def iniciar_pipeline_ml_completo(
    user_id: int,
    background_tasks: BackgroundTasks,
    ruta_dataset_externo: Optional[str] = Query(None, description="Ruta opcional a un archivo CSV externo de validación biológica"),
    db: Session = Depends(get_db)
):
    """
    Dispara en segundo plano el pipeline integral de Machine Learning buscando automáticamente 
    los archivos/transcriptomas subidos por el usuario.
    """
    try:
        # 1. Validar existencia del usuario
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"El usuario con ID {user_id} no está registrado.")

        # 2. Buscar automáticamente el último transcriptoma registrado por el usuario
        transcriptome = db.query(TranscriptomeReport).filter_by(user_id=user_id).order_by(TranscriptomeReport.id.desc()).first()
        if not transcriptome:
            raise HTTPException(status_code=404, detail=f"No se encontraron transcriptomas subidos para el usuario #{user_id}.")

        transcriptome_id = transcriptome.id

        # 3. Encolar la tarea en segundo plano
        background_tasks.add_task(
            entrenar_comparar_modelos_ml_background, 
            user_id=user_id, 
            transcriptome_id=transcriptome_id, 
            ruta_dataset_externo=ruta_dataset_externo
        )

        return {
            "message": f"Pipeline integral de ML iniciado automáticamente para el Usuario #{user_id} usando su último transcriptoma (#{transcriptome_id}).",
            "user_id": user_id,
            "transcriptome_id": transcriptome_id,
            "estado": "procesando",
            "directorio_visuales": f"data/Visuales/user_{user_id}_run_{transcriptome_id}"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al iniciar el pipeline de ML: {str(e)}")