from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.models import IdentifiedMirna, MirnaInputSummary, PipelineJobStatus, User
from app.services.database import get_db
from app.services.mirbase_service import procesar_clasificacion_desde_db_background

router = APIRouter(prefix="/mirbase", tags=["Comparación miRBase"])


@router.post("/comparar-guardados/")
async def comparar_miRNAs_guardados_endpoint(
    background_tasks: BackgroundTasks,
    user_id: int = Form(..., description="ID del usuario que realiza la comparación"),
    db: Session = Depends(get_db),
):
    try:
        user = db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail=f"El usuario con ID {user_id} no está registrado.")

        summaries = db.query(MirnaInputSummary).filter(MirnaInputSummary.user_id == user_id).all()
        if not summaries:
            raise HTTPException(
                status_code=404,
                detail=f"No hay resúmenes de miRNA (collapsed en disco) para el usuario {user_id}.",
            )

        missing = [s for s in summaries if not Path(s.collapsed_path).exists()]
        if len(missing) == len(summaries):
            raise HTTPException(
                status_code=404,
                detail="Los TSV colapsados no están en disco. Vuelve a subir/procesar la Entrada 1.",
            )

        db.query(IdentifiedMirna).filter(IdentifiedMirna.user_id == user_id).delete()
        db.commit()

        for s in summaries:
            job_key = f"clasificacion_{s.archivo}"
            job = (
                db.query(PipelineJobStatus)
                .filter_by(user_id=user_id, archivo=job_key, tipo="miRBase Classification")
                .first()
            )
            if not job:
                job = PipelineJobStatus(
                    user_id=user_id,
                    archivo=job_key,
                    tipo="miRBase Classification",
                    estado="procesando",
                )
                db.add(job)
            else:
                job.estado = "procesando"
        db.commit()

        background_tasks.add_task(procesar_clasificacion_desde_db_background, user_id)

        return {
            "message": f"Clasificación miRBase iniciada para usuario {user_id}.",
            "user_id": user_id,
            "estado": "procesando",
            "referencia_utilizada": "mature.fa",
            "collapsed_files": [s.collapsed_path for s in summaries if Path(s.collapsed_path).exists()],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


@router.get("/estadisticas-clasificacion/{user_id}")
def obtener_estadisticas_clasificacion(user_id: int, db: Session = Depends(get_db)):
    jobs_activos = (
        db.query(PipelineJobStatus)
        .filter(
            PipelineJobStatus.user_id == user_id,
            PipelineJobStatus.tipo == "miRBase Classification",
            PipelineJobStatus.estado == "procesando",
        )
        .count()
    )

    if jobs_activos > 0:
        return {
            "status": "cargando",
            "user_id": user_id,
            "mensaje": "Clasificación miRBase en segundo plano...",
            "total_registros_analizados": 0,
            "indicadores_clave": {},
            "resumen_por_categoria": [],
            "tabla_muestra_resultados": [],
        }

    total = db.query(IdentifiedMirna).filter(IdentifiedMirna.user_id == user_id).count()
    if total == 0:
        return {
            "status": "pendiente",
            "user_id": user_id,
            "mensaje": f"Aún no hay miRNAs identificados para el usuario {user_id}.",
            "total_registros_analizados": 0,
            "indicadores_clave": {},
            "resumen_por_categoria": [],
            "tabla_muestra_resultados": [],
        }

    conteo = (
        db.query(IdentifiedMirna.match_type, func.count(IdentifiedMirna.id))
        .filter(IdentifiedMirna.user_id == user_id)
        .group_by(IdentifiedMirna.match_type)
        .all()
    )

    resumen = []
    conocidos = 0
    novel = 0
    for match_type, cantidad in conteo:
        pct = round((cantidad / total) * 100, 2)
        resumen.append({"clasificacion": match_type, "cantidad": cantidad, "porcentaje": pct})
        if match_type == "novel":
            novel += cantidad
        else:
            conocidos += cantidad

    muestra = db.query(IdentifiedMirna).filter(IdentifiedMirna.user_id == user_id).limit(50).all()

    return {
        "status": "completado",
        "user_id": user_id,
        "total_registros_analizados": total,
        "indicadores_clave": {
            "total_conocidos": conocidos,
            "porcentaje_conocidos": round((conocidos / total) * 100, 2),
            "total_novedosos": novel,
            "porcentaje_novedosos": round((novel / total) * 100, 2),
        },
        "resumen_por_categoria": resumen,
        "tabla_muestra_resultados": [
            {
                "source_file": m.source_file,
                "mirna_id": m.mirna_id,
                "mirbase_id": m.mirbase_id,
                "sequence": m.mirna_sequence,
                "species": m.species,
                "read_count": m.read_count,
                "match_type": m.match_type,
                "mismatches": m.mismatches,
                "alignment_score": m.alignment_score,
            }
            for m in muestra
        ],
    }
