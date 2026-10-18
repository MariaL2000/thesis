from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from sqlalchemy.orm import Session

from app.models.models import (
    IdentifiedMirna,
    InteractionCandidate,
    PipelineJobStatus,
    TranscriptomeReport,
    User,
)
from app.services.database import get_db
from app.services.interaction_service import procesar_interacciones_background

router = APIRouter(prefix="/interaction", tags=["Predicción y Características de Interacción"])


@router.post("/calcular-interacciones")
async def calcular_interacciones_endpoint(
    background_tasks: BackgroundTasks,
    user_id: int = Form(..., description="ID del usuario que solicita el cálculo"),
    max_pares_por_mirna: int = Form(5, description="Máximo de blancos mRNA por miRNA"),
    db: Session = Depends(get_db),
):
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
            detail=f"No hay transcriptoma para el usuario {user_id}.",
        )

    mirna_count = db.query(IdentifiedMirna).filter_by(user_id=user_id).count()
    if mirna_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No hay miRNAs identificados. Ejecuta antes POST /mirbase/comparar-guardados/.",
        )

    # Registrar job ANTES del background para que el GET no se quede en 'cargando' eterno
    job_nombre = f"Prediction Interaction - Run #{transcriptome.id}"
    job = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
    if not job:
        job = PipelineJobStatus(
            user_id=user_id,
            archivo=f"transcriptome_{transcriptome.id}",
            tipo=job_nombre,
            estado="procesando",
        )
        db.add(job)
    else:
        job.estado = "procesando"
        job.detalle_error = None
    db.commit()

    background_tasks.add_task(
        procesar_interacciones_background, user_id, transcriptome.id, max_pares_por_mirna
    )

    return {
        "status": "procesando",
        "user_id": user_id,
        "transcriptome_id": transcriptome.id,
        "archivo": transcriptome.archivo,
        "mirnas_disponibles": mirna_count,
        "message": f"Cálculo de interacciones iniciado para '{transcriptome.archivo}'.",
        "status_get": f"/api/v1/interaction/estadisticas-interaccion/{user_id}",
    }


@router.get("/estadisticas-interaccion/{user_id}")
async def obtener_estado_interacciones_por_analisis(user_id: int, db: Session = Depends(get_db)):
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
        raise HTTPException(status_code=404, detail=f"No hay transcriptoma para el usuario {user_id}.")

    transcriptome_id = transcriptome.id
    job_nombre = f"Prediction Interaction - Run #{transcriptome_id}"
    job = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()

    if not job or job.estado == "procesando":
        return {
            "status": "cargando",
            "user_id": user_id,
            "transcriptome_id": transcriptome_id,
            "mensaje": "Cálculo de interacciones en segundo plano...",
        }

    if job.estado == "error":
        return {
            "status": "error",
            "user_id": user_id,
            "transcriptome_id": transcriptome_id,
            "mensaje": job.detalle_error,
        }

    total = (
        db.query(InteractionCandidate)
        .filter_by(user_id=user_id, transcriptome_id=transcriptome_id)
        .count()
    )
    labeled_neg = (
        db.query(InteractionCandidate)
        .filter_by(
            user_id=user_id,
            transcriptome_id=transcriptome_id,
            label_source="synthetic_negative",
        )
        .count()
    )
    unlabeled = (
        db.query(InteractionCandidate)
        .filter_by(user_id=user_id, transcriptome_id=transcriptome_id, label_source="unlabeled")
        .count()
    )

    muestra = (
        db.query(InteractionCandidate)
        .filter_by(user_id=user_id, transcriptome_id=transcriptome_id)
        .limit(15)
        .all()
    )

    return {
        "status": "completado",
        "user_id": user_id,
        "transcriptome_id": transcriptome_id,
        "resumen_estadistico": {
            "total_candidatos": total,
            "synthetic_negatives": labeled_neg,
            "unlabeled": unlabeled,
            "identified_mirnas": db.query(IdentifiedMirna).filter_by(user_id=user_id).count(),
        },
        "dataset_listo_para_ml": [
            {
                "mirna_id": i.mirna_id,
                "mrna_id": i.mrna_id,
                "gene_id": i.gene_id,
                "complementarity": i.complementarity,
                "seed_match": i.seed_match,
                "seed_position": i.seed_position,
                "mismatches": i.mismatches,
                "gaps": i.gaps,
                "delta_g": i.delta_g,
                "accessibility": i.accessibility,
                "mirna_expression": i.mirna_expression,
                "label": i.label,
                "label_source": i.label_source,
                "ml_probability": i.ml_probability,
            }
            for i in muestra
        ],
    }
