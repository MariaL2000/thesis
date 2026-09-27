from fastapi import APIRouter, Form, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from app.services.interaction_service import procesar_interacciones_background
from app.services.database import get_db
from app.models.models import InteractionReport, PipelineJobStatus, TranscriptomeReport, User, MirnaClassificationReport

router = APIRouter(prefix="/interaction", tags=["Predicción y Características de Interacción"])

@router.post("/calcular-interacciones")
async def calcular_interacciones_endpoint(
    background_tasks: BackgroundTasks,
    user_id: int = Form(..., description="ID del usuario que solicita el cálculo"),
    max_pares_por_mirna: int = Form(5, description="Límite máximo de blancos (mRNA) a evaluar por cada miRNA"),
    db: Session = Depends(get_db)
):
    """Dispara el análisis comparativo (conocidos + novedosos) seleccionando automáticamente el transcriptoma más reciente del usuario."""
    
    # 1. Validar que el usuario exista
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"El usuario con ID {user_id} no está registrado.")

    # 2. Buscar el transcriptoma más reciente del usuario de forma automática
    transcriptome = (
        db.query(TranscriptomeReport)
        .filter_by(user_id=user_id)
        .order_by(TranscriptomeReport.id.desc()) 
        .first()
    )
    
    if not transcriptome:
        raise HTTPException(
            status_code=404, 
            detail=f"No se encontró ningún archivo de transcriptoma registrado para el usuario {user_id}. Por favor, sube un archivo primero."
        )

    transcriptome_id = transcriptome.id
    archivo_a_procesar = transcriptome.archivo

    try:
        # 3. Lanzar la tarea en segundo plano pasando el ID detectado automáticamente
        background_tasks.add_task(procesar_interacciones_background, user_id, transcriptome_id, max_pares_por_mirna)
        
        return {
            "status": "procesando",
            "user_id": user_id,
            "transcriptome_id": transcriptome_id,
            "archivo": archivo_a_procesar,
            "message": f"Cálculo comparativo (conocidos + novedosos) iniciado automáticamente para tu archivo más reciente: '{archivo_a_procesar}' (ID: {transcriptome_id})."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/estadisticas-interaccion/{user_id}")
async def obtener_estado_interacciones_por_analisis(user_id: int, db: Session = Depends(get_db)):
    """Consulta el estado, los resultados de interacciones y la comparativa entre miRNAs conocidos y novedosos para el transcriptoma más reciente."""
    
    # 1. Validar que el usuario exista
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"El usuario con ID {user_id} no está registrado.")

    # 2. Buscar automáticamente el transcriptome_id más reciente del usuario
    transcriptome = (
        db.query(TranscriptomeReport)
        .filter_by(user_id=user_id)
        .order_by(TranscriptomeReport.id.desc())
        .first()
    )
    
    if not transcriptome:
        raise HTTPException(
            status_code=404, 
            detail=f"No se encontró ningún transcriptoma registrado para el usuario {user_id}."
        )

    transcriptome_id = transcriptome.id
    job_nombre = f"Prediction Interaction - Run #{transcriptome_id}"
    
    # 3. Filtrar el estado del job por usuario y nombre específico del último análisis
    job = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
    
    if not job or job.estado == "procesando":
        return {
            "status": "cargando",
            "user_id": user_id,
            "transcriptome_id": transcriptome_id,
            "mensaje": "El cálculo de interacciones comparativas sigue ejecutándose en segundo plano..."
        }
    
    if job.estado == "error":
        return {
            "status": "error",
            "user_id": user_id,
            "transcriptome_id": transcriptome_id,
            "mensaje": job.detalle_error
        }

    # 4. Obtener estadísticas globales y desagregadas por tipo de miRNA
    total_interacciones = db.query(InteractionReport).filter_by(user_id=user_id, transcriptome_id=transcriptome_id).count()
    
    # Realizamos un join o cruce rápido con los miRNAs clasificados para enriquecer las estadísticas de salida
    interacciones_con_clasificacion = (
        db.query(InteractionReport, MirnaClassificationReport.clasificacion)
        .join(
            MirnaClassificationReport,
            (InteractionReport.user_id == MirnaClassificationReport.user_id) & 
            (InteractionReport.miRNA == MirnaClassificationReport.miRNA)
        )
        .filter(InteractionReport.user_id == user_id, InteractionReport.transcriptome_id == transcriptome_id)
        .all()
    )

    # Conteo dinámico por categoría
    conteo_conocidos = sum(1 for i, clasif in interacciones_con_clasificacion if "conocido" in clasif.lower())
    conteo_novedosos = sum(1 for i, clasif in interacciones_con_clasificacion if "novedoso" in clasif.lower() or "novel" in clasif.lower())

    # Obtener una muestra representativa (hasta 15 elementos)
    muestra = db.query(InteractionReport).filter_by(user_id=user_id, transcriptome_id=transcriptome_id).limit(15).all()

    return {
        "status": "completado",
        "user_id": user_id,
        "transcriptome_id": transcriptome_id,
        "resumen_estadistico": {
            "total_interacciones_registradas": total_interacciones,
            "interacciones_miRNAs_conocidos": conteo_conocidos,
            "interacciones_miRNAs_novedosos": conteo_novedosos
        },
        "dataset_listo_para_ml": [
            {
                "miRNA": i.miRNA,
                "mRNA": i.mRNA,
                "Complementariedad": i.complementariedad,
                "Seed_Match": i.seed_match,
                "Mismatches": i.mismatches,
                "Delta_G": i.delta_g,
                "Accesibilidad": i.accesibilidad,
                "Interaction_Score": i.interaction_score
            } for i in muestra
        ]
    }