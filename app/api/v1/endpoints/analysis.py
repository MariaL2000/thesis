from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import DEFAULT_MAX_LEN_MIRNA, DEFAULT_MIN_LEN_MIRNA
from app.models.models import (
    InputFile,
    MirnaInputSummary,
    PipelineJobStatus,
    TranscriptomeReport,
    User,
    ValidationReport,
)
from app.schemas.schemas import InputSummaryResponse, PipelineUploadResponse, ValidationStatusResponse
from app.services.database import get_db
from app.services.inputs_service import (
    check_local_mirbase,
    peek_validacion_estructura,
    procesar_y_guardar_muestreo_input,
    validar_y_guardar_archivos_obligatorios,
)

router = APIRouter()


def tarea_procesamiento_background(
    user_id: int,
    paths_y_tipos: dict,
    min_len: int,
    max_len: int,
):
    """1) Peek de validación (GET temprano) → 2) QC/colapsado completo."""
    print(f"[BACKGROUND] Procesamiento user_id={user_id}")
    for tipo_archivo, path_archivo in paths_y_tipos.items():
        role = "mirna" if "miRNA" in tipo_archivo else "transcriptome"
        try:
            peek = peek_validacion_estructura(
                path_archivo,
                role=role,
                user_id=user_id,
                min_len_mirna=min_len,
                max_len_mirna=max_len,
            )
            print(f"[BACKGROUND] Peek {path_archivo}: {peek}")
            if peek.get("status") == "error":
                print(f"[BACKGROUND] Validación fallida, se omite QC completo de {path_archivo}")
                continue

            resultado = procesar_y_guardar_muestreo_input(
                path_archivo,
                tipo_archivo,
                user_id=user_id,
                min_len_mirna=min_len,
                max_len_mirna=max_len,
            )
            print(f"[BACKGROUND] OK {path_archivo}: {resultado}")
        except Exception as e:
            print(f"[BACKGROUND] Error {path_archivo}: {e}")
    print(f"[BACKGROUND] Finalizado user={user_id}")


@router.post("/upload-inputs/", response_model=PipelineUploadResponse)
async def upload_inputs(
    background_tasks: BackgroundTasks,
    user_id: int = Form(0, description="ID del usuario (si es 0, se creará automáticamente)"),
    min_len_mirna: int = Form(DEFAULT_MIN_LEN_MIRNA, description="Longitud mínima miRNA (default 18)"),
    max_len_mirna: int = Form(
        DEFAULT_MAX_LEN_MIRNA,
        description="Longitud máxima miRNA (default 25; ampliar a 30 para plantas si hace falta)",
    ),
    mirna_file: UploadFile = File(
        ..., description="[OBLIGATORIO] miRNA / small RNA-seq (FASTA/FASTQ/GZ, máx 1GB)"
    ),
    transcriptome_file: UploadFile = File(
        ..., description="[OBLIGATORIO] transcriptoma / mRNA (FASTA/FASTQ/GZ, máx 1GB)"
    ),
    db: Session = Depends(get_db),
):
    try:
        user = None
        if user_id > 0:
            user = db.query(User).filter_by(id=user_id).first()

        if not user:
            new_username = f"user_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
            user = User(username=new_username)
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id

        paths_y_tipos = validar_y_guardar_archivos_obligatorios(
            mirna_file, transcriptome_file, user_id=user_id
        )

        background_tasks.add_task(
            tarea_procesamiento_background,
            user_id,
            paths_y_tipos,
            min_len_mirna,
            max_len_mirna,
        )

        return PipelineUploadResponse(
            message=f"Entradas recibidas. Usuario {user_id} procesando en segundo plano.",
            user_id=user_id,
            archivos_en_proceso=list(paths_y_tipos.values()),
            rango_mirna_aplicado={"min": min_len_mirna, "max": max_len_mirna},
            storage_policy="sequences_on_disk_metadata_in_db",
            mirbase_status=check_local_mirbase("mature.fa"),
            validation_get=f"/api/v1/validation-status/{user_id}",
            input_summary_get=f"/api/v1/input-summary/{user_id}",
        )

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error al procesar las entradas: {str(e)}")


@router.get("/validation-status/{user_id}", response_model=ValidationStatusResponse)
def get_validation_status(user_id: int, db: Session = Depends(get_db)):
    """GET de validación/QC. Consultable mientras corre el background y al terminar."""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado.")

    reports = (
        db.query(ValidationReport)
        .filter_by(user_id=user_id)
        .order_by(ValidationReport.id.desc())
        .limit(10)
        .all()
    )
    jobs = db.query(PipelineJobStatus).filter_by(user_id=user_id).all()

    mirna_sum = (
        db.query(MirnaInputSummary)
        .filter_by(user_id=user_id)
        .order_by(MirnaInputSummary.id.desc())
        .first()
    )
    tr_sum = (
        db.query(TranscriptomeReport)
        .filter_by(user_id=user_id)
        .order_by(TranscriptomeReport.id.desc())
        .first()
    )

    if not reports:
        return ValidationStatusResponse(
            status="pendiente",
            user_id=user_id,
            mensaje="Aún no hay reportes de validación. Sube archivos en POST /upload-inputs/.",
            next_steps=["POST /api/v1/upload-inputs/ con mirna_file y transcriptome_file"],
        )

    pending = any(r.status in ("procesando", "validando") for r in reports)
    has_error = any(r.status == "error" for r in reports)
    both_done = (
        mirna_sum is not None
        and tr_sum is not None
        and all(r.status == "completado" for r in reports[:2])
    )

    if pending:
        status = "procesando"
        mensaje = "Validación/QC en segundo plano. Vuelve a consultar este GET."
        next_steps = [f"GET /api/v1/validation-status/{user_id}"]
    elif has_error and not both_done:
        status = "error"
        mensaje = "Al menos una entrada falló la validación. Revisa 'validaciones.detail'."
        next_steps = ["Corrige los archivos y vuelve a subir con POST /upload-inputs/"]
    elif both_done:
        status = "completado"
        mensaje = "Ambas entradas validadas y procesadas. Puedes continuar con miRBase."
        next_steps = [
            f"GET /api/v1/input-summary/{user_id}",
            "POST /api/v1/mirbase/comparar-guardados/",
            f"GET /api/v1/mirbase/estadisticas-clasificacion/{user_id}",
        ]
    else:
        status = "parcial"
        mensaje = "Proceso parcial: falta completar alguna entrada."
        next_steps = [f"GET /api/v1/validation-status/{user_id}"]

    return ValidationStatusResponse(
        status=status,
        user_id=user_id,
        mensaje=mensaje,
        validaciones=[
            {
                "role": r.role,
                "filename": r.filename,
                "status": r.status,
                "format_ok": r.format_ok,
                "size_ok": r.size_ok,
                "structure_ok": r.structure_ok,
                "looks_like_mirna": r.looks_like_mirna,
                "total_records": r.total_records_sampled,
                "invalid_records": r.invalid_records,
                "avg_quality": r.avg_quality,
                "detail": r.detail,
            }
            for r in reports
        ],
        jobs=[
            {
                "archivo": j.archivo,
                "tipo": j.tipo,
                "estado": j.estado,
                "total_leidos": j.total_leidos,
                "validos_guardados": j.validos_guardados,
                "descartados": j.descartados,
                "error": j.detalle_error,
            }
            for j in jobs
        ],
        mirna_summary=(
            {
                "archivo": mirna_sum.archivo,
                "collapsed_path": mirna_sum.collapsed_path,
                "total_reads": mirna_sum.total_reads,
                "reads_after_qc": mirna_sum.reads_after_qc,
                "unique_sequences": mirna_sum.unique_sequences,
                "discarded": mirna_sum.discarded,
                "min_len": mirna_sum.min_len,
                "max_len": mirna_sum.max_len,
            }
            if mirna_sum
            else None
        ),
        transcriptome_summary=(
            {
                "id": tr_sum.id,
                "archivo": tr_sum.archivo,
                "total_transcritos": tr_sum.total_transcritos,
                "transcritos_validos": tr_sum.transcritos_validos,
                "longitud_minima_bp": tr_sum.longitud_minima_bp,
                "longitud_maxima_bp": tr_sum.longitud_maxima_bp,
                "longitud_promedio_bp": tr_sum.longitud_promedio_bp,
                "index_path": tr_sum.index_path,
            }
            if tr_sum
            else None
        ),
        next_steps=next_steps,
    )


@router.get("/input-summary/{user_id}", response_model=InputSummaryResponse)
def get_input_summary(user_id: int, db: Session = Depends(get_db)):
    """GET con metadatos de archivos + resúmenes de QC (sin secuencias)."""
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"Usuario {user_id} no encontrado.")

    files = (
        db.query(InputFile)
        .filter_by(user_id=user_id)
        .order_by(InputFile.id.desc())
        .limit(20)
        .all()
    )
    mirnas = (
        db.query(MirnaInputSummary)
        .filter_by(user_id=user_id)
        .order_by(MirnaInputSummary.id.desc())
        .all()
    )
    transcriptomes = (
        db.query(TranscriptomeReport)
        .filter_by(user_id=user_id)
        .order_by(TranscriptomeReport.id.desc())
        .all()
    )

    ready = bool(mirnas) and bool(transcriptomes)
    return InputSummaryResponse(
        status="completado" if ready else ("parcial" if files else "pendiente"),
        user_id=user_id,
        mensaje=(
            "Entradas listas para miRBase / interacciones."
            if ready
            else "Esperando que termine el QC de ambas entradas."
        ),
        input_files=[
            {
                "id": f.id,
                "role": f.role,
                "filename": f.original_filename,
                "format": f.format,
                "size_bytes": f.size_bytes,
                "stored_path": f.stored_path,
            }
            for f in files
        ],
        mirna_summaries=[
            {
                "archivo": m.archivo,
                "collapsed_path": m.collapsed_path,
                "total_reads": m.total_reads,
                "reads_after_qc": m.reads_after_qc,
                "unique_sequences": m.unique_sequences,
                "discarded": m.discarded,
                "min_len": m.min_len,
                "max_len": m.max_len,
            }
            for m in mirnas
        ],
        transcriptome_summaries=[
            {
                "id": t.id,
                "archivo": t.archivo,
                "total_transcritos": t.total_transcritos,
                "transcritos_validos": t.transcritos_validos,
                "longitud_minima_bp": t.longitud_minima_bp,
                "longitud_maxima_bp": t.longitud_maxima_bp,
                "longitud_promedio_bp": t.longitud_promedio_bp,
                "index_path": t.index_path,
            }
            for t in transcriptomes
        ],
        validation_get=f"/api/v1/validation-status/{user_id}",
    )
