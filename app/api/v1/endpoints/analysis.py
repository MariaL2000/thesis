from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Form, Depends
from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from app.models.models import PipelineJobStatus, User, TranscriptomeReport
from app.services.database import get_db
from app.services.inputs_service import (
    validar_y_guardar_archivos_obligatorios,
    procesar_y_guardar_muestreo_input,
    check_local_mirbase,
    DEFAULT_MAX_RECORDS_SAMPLE
)

router = APIRouter()


def tarea_procesamiento_background(
    user_id: int, 
    paths_y_tipos: dict, 
    max_records: int, 
    min_len: int, 
    max_len: int
):
    print("\n" + "🌀 " * 20)
    print(f"⏳ [BACKGROUND] Tarea de procesamiento para el usuario ID: {user_id} iniciada...")
    print("🌀 " * 20)

    for tipo_archivo, path_archivo in paths_y_tipos.items():
        print(f"\n📂 [Background] Analizando archivo: Path={path_archivo} | Tipo={tipo_archivo} | UserID={user_id}")
        try:
            resultado = procesar_y_guardar_muestreo_input(
                path_archivo,
                tipo_archivo,
                user_id=user_id,
                max_records=max_records,
                min_len_mirna=min_len,
                max_len_mirna=max_len
            )
            print(f"✅ [Background] Éxito en {path_archivo}:")
            for k, v in resultado.items():
                print(f"   - {k}: {v}")
        except Exception as e:
            print(f"❌ [Background] Error crítico procesando {path_archivo}: {str(e)}")

    print("\n" + "🏁 " * 20)
    print(f"🎉 [BACKGROUND] ¡Procesamiento masivo finalizado con éxito para el usuario {user_id}!")
    print("🏁 " * 20 + "\n")


@router.post("/upload-inputs/")
async def upload_inputs(
    background_tasks: BackgroundTasks,
    user_id: int = Form(0, description="ID del usuario (si es 0, se creará automáticamente)"),
    max_records: int = Form(DEFAULT_MAX_RECORDS_SAMPLE),
    min_len_mirna: int = Form(18, description="Longitud mínima de miRNA (por defecto 18 nt)"),
    max_len_mirna: int = Form(25, description="Longitud máxima de miRNA (por defecto 25 nt, ampliable a 40-50)"),
    mirna_file: UploadFile = File(..., description="[OBLIGATORIO] Archivo único de secuencias miRNA o small RNA-seq (FASTA/FASTQ/GZ, máx 1GB)"),
    transcriptome_file: UploadFile = File(..., description="[OBLIGATORIO] Archivo único de transcriptoma o mRNA (FASTA/FASTQ/GZ, máx 1GB)"),
    db: Session = Depends(get_db)
):
    try:
        # 1. Autorregistro o verificación de usuario
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

        # 2. Validar que ambas entradas obligatorias existan y no superen 1 GB por archivo
        paths_y_tipos = validar_y_guardar_archivos_obligatorios(
            mirna_file, 
            transcriptome_file, 
            user_id=user_id
        )

        # 3. Enviar a segundo plano
        background_tasks.add_task(
            tarea_procesamiento_background,
            user_id,
            paths_y_tipos,
            max_records,
            min_len_mirna,
            max_len_mirna
        )

        mirbase_check = check_local_mirbase("mature.fa")

        return {
            "message": f"Las dos entradas obligatorias se recibieron correctamente. Usuario ID {user_id} procesando en segundo plano.",
            "user_id": user_id,
            "archivos_en_proceso": list(paths_y_tipos.values()),
            "rango_mirna_aplicado": {"min": min_len_mirna, "max": max_len_mirna},
            "mirbase_status": mirbase_check
        }

    except HTTPException as he:
        raise he
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error al procesar las entradas: {str(e)}")