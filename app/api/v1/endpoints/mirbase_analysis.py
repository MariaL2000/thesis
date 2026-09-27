from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Form
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.services.database import get_db
from app.models.models import MirnaReport, MirnaClassificationReport, PipelineJobStatus, User
from app.services.mirbase_service import procesar_clasificacion_desde_db_background

router = APIRouter(prefix="/mirbase", tags=["Comparación miRBase"])

UPLOAD_DIR = Path("data/Secuencias")

@router.post("/comparar-guardados/")
async def comparar_miRNAs_guardados_endpoint(
    background_tasks: BackgroundTasks,
    user_id: int = Form(..., description="ID del usuario que realiza la comparación"),
    db: Session = Depends(get_db)
):
    """
    Endpoint optimizado: Consulta la BD para un usuario específico, valida archivos, 
    registra el estado de carga y lanza la clasificación contra miRBase en segundo plano.
    """
    try:
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"El usuario con ID {user_id} no está registrado.")

        mirna_reports = db.query(MirnaReport).filter(MirnaReport.user_id == user_id).all()
        if not mirna_reports:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron registros previos de miRNAs para el usuario {user_id} en la base de datos."
            )

        user_dir = UPLOAD_DIR / f"user_{user_id}"
        nombres_archivos = sorted(list(set(r.archivo for r in mirna_reports if (user_dir / r.archivo).exists())))
        
        if not nombres_archivos:
            raise HTTPException(
                status_code=404,
                detail=f"Los registros existen en la BD pero los archivos físicos no están disponibles en el disco para el usuario {user_id}."
            )

        # Limpiar reportes de clasificación anteriores de este usuario
        db.query(MirnaClassificationReport).filter(MirnaClassificationReport.user_id == user_id).delete()
        db.commit()

        # Registrar o actualizar el estado en PipelineJobStatus por usuario
        for archivo in nombres_archivos:
            job_key = f"clasificacion_{archivo}"
            job = db.query(PipelineJobStatus).filter_by(user_id=user_id, archivo=job_key, tipo="miRBase Classification").first()
            if not job:
                job = PipelineJobStatus(user_id=user_id, archivo=job_key, tipo="miRBase Classification", estado="procesando")
                db.add(job)
            else:
                job.estado = "procesando"
        db.commit()

        # Lanzar el procesamiento en segundo plano
        background_tasks.add_task(procesar_clasificacion_desde_db_background, user_id)

        return {
            "message": f"Análisis comparativo con miRBase iniciado en segundo plano para el usuario {user_id}.",
            "user_id": user_id,
            "estado": "procesando",
            "referencia_utilizada": "mature.fa",
            "archivos_unicos_a_procesar": nombres_archivos
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/estadisticas-clasificacion/{user_id}")
def obtener_estadisticas_clasificacion(user_id: int, db: Session = Depends(get_db)):
    """
    Verifica si hay procesos activos para el usuario. Si está en curso, avisa que sigue cargando.
    Si terminó, calcula estadísticas globales y desglosa los resultados conocidos y novedosos.
    """
    jobs_activos = db.query(PipelineJobStatus).filter(
        PipelineJobStatus.user_id == user_id,
        PipelineJobStatus.tipo == "miRBase Classification",
        PipelineJobStatus.estado == "procesando"
    ).count()

    if jobs_activos > 0:
        return {
            "status": "cargando",
            "user_id": user_id,
            "mensaje": "El análisis comparativo con miRBase está en ejecución en segundo plano. Por favor, espere...",
            "total_registros_analizados": 0,
            "indicadores_clave": {},
            "resumen_por_categoria": [],
            "tabla_muestra_resultados": []
        }

    ultima_fecha_sesion = db.query(func.max(MirnaClassificationReport.created_at)).filter(
        MirnaClassificationReport.user_id == user_id
    ).scalar()
    
    if not ultima_fecha_sesion:
        return {
            "status": "pendiente",
            "user_id": user_id,
            "mensaje": f"Aún no hay resultados de clasificación en la base de datos para el usuario {user_id}.",
            "total_registros_analizados": 0,
            "indicadores_clave": {},
            "resumen_por_categoria": [],
            "tabla_muestra_resultados": []
        }

    query_sesion_actual = db.query(MirnaClassificationReport).filter(
        MirnaClassificationReport.user_id == user_id,
        MirnaClassificationReport.created_at == ultima_fecha_sesion
    )
    
    total_registros = query_sesion_actual.count()
    
    if total_registros == 0:
        return {
            "status": "pendiente",
            "user_id": user_id,
            "mensaje": "No se encontraron registros para la última sesión del usuario.",
            "total_registros_analizados": 0,
            "indicadores_clave": {},
            "resumen_por_categoria": [],
            "tabla_muestra_resultados": []
        }

    conteo_clasificacion = db.query(
        MirnaClassificationReport.clasificacion, 
        func.count(MirnaClassificationReport.id).label("cantidad")
    ).filter(
        MirnaClassificationReport.user_id == user_id,
        MirnaClassificationReport.created_at == ultima_fecha_sesion
    ).group_by(MirnaClassificationReport.clasificacion).all()

    estadisticas_detalladas = []
    total_novedosos = 0
    total_conocidos = 0

    for clasif, cantidad in conteo_clasificacion:
        porcentaje = round((cantidad / total_registros) * 100, 2) if total_registros > 0 else 0
        estadisticas_detalladas.append({
            "clasificacion": clasif,
            "cantidad": cantidad,
            "porcentaje": porcentaje
        })
        
        if "Novedoso" in clasif or "Novel" in clasif:
            total_novedosos += cantidad
        else:
            total_conocidos += cantidad

    muestra_datos = query_sesion_actual.limit(50).all()
    tabla_muestra = [
        {
            "archivo": m.archivo,
            "miRNA": m.miRNA,
            "secuencia_completa": m.secuencia,
            "secuencia_parcial": (m.secuencia[:15] + "...") if m.secuencia and len(m.secuencia) > 15 else m.secuencia,
            "clasificacion": m.clasificacion
        }
        for m in muestra_datos
    ]

    return {
        "status": "completado",
        "user_id": user_id,
        "total_registros_analizados": total_registros,
        "indicadores_clave": {
            "total_conocidos": total_conocidos,
            "porcentaje_conocidos": round((total_conocidos / total_registros) * 100, 2) if total_registros > 0 else 0,
            "total_novedosos": total_novedosos,
            "porcentaje_novedosos": round((total_novedosos / total_registros) * 100, 2) if total_registros > 0 else 0
        },
        "resumen_por_categoria": estadisticas_detalladas,
        "tabla_muestra_resultados": tabla_muestra
    }


"""
Interpretación de las Proporciones (86.45% Novedosos vs. 13.55% Conocidos) corrindo el endpoint 
El sesgo hacia lo novedoso es habitual en bioinformática: Es muy común que una corrida analítica inicial o un conjunto filtrado de secuencias arroje un porcentaje alto de candidatos catalogados como "Potencialmente Novedosos" (Novel miRNA).

Factores que explican esta proporción:

Criterio de búsqueda (Homología por región seed): El resultado indica que los conocidos se detectaron por homología en la región semilla (seed region, nucleótidos 2-8). Si el organismo bajo estudio no es un modelo altamente anotado (como humano o ratón) o si miRBase no cuenta con una representación exhaustiva de ese clado/especie, muchas secuencias legítimas no encontrarán un match exacto con los miRNAs ya publicados.

Inclusión de artefactos o fragmentos de otros ARNs: En los archivos crudos de pequeños ARN (sRNA-seq), la gran mayoría de las lecturas que no mapean contra miRNAs conocidos de bases de datos globales suelen ser fragmentos de ARNt (tRFs), ARNr, ARN sno, u otras moléculas de ARN no codificante que los algoritmos de predicción inicial pueden clasificar provisionalmente como candidatos a novel miRNAs debido a su tamaño o características estructurales.

2. Análisis de la Tabla de Muestra (tabla_muestra_resultados)
Nomenclatura (_freq_1): Los identificadores muestran sufijos como SRR2857730.249115_freq_1. Esto indica que son secuencias únicas (frecuencia = 1) obtenidas tras el colapso de lecturas redundantes (collapsed reads), un paso estándar en pipelines de análisis de pequeños ARNs para optimizar la computación.

Longitudes de las secuencias: Las secuencias mostradas oscilan típicamente entre los 20 y los 25 nucleótidos, que es precisamente el rango de tamaño biológico natural de los microRNAs maduros. Esto confirma que el filtrado de tamaño de la librería funcionó de manera correcta.
"""