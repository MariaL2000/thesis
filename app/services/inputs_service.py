import gzip
import statistics
from pathlib import Path
from typing import Dict, Any
from Bio import SeqIO
from fastapi import UploadFile, HTTPException

from app.services.database import SessionLocal
from app.models.models import MirnaReport, PipelineJobStatus, TranscriptomeReport

UPLOAD_DIR = Path("data/Secuencias")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MAX_RECORDS_SAMPLE = 30000
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB por archivo

# Adaptador completo de 3' para Illumina Small RNA (TruSeq)
ADAPTADOR_ILLUMINA = "TGGAATTCTCGGGTGCCAAGG"

def limpiar_adaptador_si_existe(secuencia: str) -> str:
    """Remueve el adaptador completo de Illumina Small RNA si está presente."""
    if ADAPTADOR_ILLUMINA in secuencia:
        idx = secuencia.find(ADAPTADOR_ILLUMINA)
        if idx >= 16:  # Deja un miRNA mínimo válido antes del adaptador
            return secuencia[:idx]
    else:
        # Opcional: permitir coincidencia parcial si el final de la lectura trunca el adaptador
        for i in range(len(ADAPTADOR_ILLUMINA), 6, -1):
            sub_adaptador = ADAPTADOR_ILLUMINA[:i]
            if secuencia.endswith(sub_adaptador) and len(secuencia) - i >= 16:
                return secuencia[:-i]
                
    return secuencia


def validar_y_guardar_archivos_obligatorios(mirna_file: UploadFile, transcriptome_file: UploadFile, user_id: int) -> dict:
    """
    Valida que ambas entradas obligatorias estén presentes, que cada una 
    no supere 1 GB por separado, y las guarda en una carpeta aislada por usuario.
    """
    if not mirna_file or not mirna_file.filename or not mirna_file.filename.strip():
        raise HTTPException(status_code=400, detail="La entrada de miRNA o small RNA-seq es obligatoria.")
    
    if not transcriptome_file or not transcriptome_file.filename or not transcriptome_file.filename.strip():
        raise HTTPException(status_code=400, detail="La entrada de transcriptoma (mRNA) es obligatoria.")

    # Validar tamaño de cada archivo individualmente (máx 1 GB c/u)
    archivos_a_validar = [
        (mirna_file, "miRNA experimental"),
        (transcriptome_file, "transcriptoma de referencia")
    ]

    user_dir = UPLOAD_DIR / f"user_{user_id}"
    user_dir.mkdir(parents=True, exist_ok=True)

    paths_y_tipos = {}

    for file_obj, tipo_archivo in archivos_a_validar:
        file_obj.file.seek(0, 2)
        tamano_bytes = file_obj.file.tell()
        file_obj.file.seek(0)

        if tamano_bytes > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"El archivo '{file_obj.filename}' ({tipo_archivo}) pesa {tamano_bytes / (1024*1024):.2f} MB y excede el límite de 1 GB."
            )

        ruta_destino = user_dir / file_obj.filename
        with open(ruta_destino, "wb") as buffer:
            while chunk := file_obj.file.read(1024 * 1024):
                buffer.write(chunk)
        
        paths_y_tipos[tipo_archivo] = str(ruta_destino)

    return paths_y_tipos


def procesar_y_guardar_muestreo_input(
    file_path: str, 
    tipo_archivo: str, 
    user_id: int, 
    max_records: int = DEFAULT_MAX_RECORDS_SAMPLE,
    min_len_mirna: int = 18,
    max_len_mirna: int = 25
) -> Dict[str, Any]:
    path_obj = Path(file_path)
    filename_lower = path_obj.name.lower()
    is_gz = filename_lower.endswith(".gz")
    open_func = gzip.open if is_gz else open
    is_mirna = "miRNA" in tipo_archivo

    db = SessionLocal()
    
    try:
        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, archivo=path_obj.name, tipo=tipo_archivo).first()
        if not job_status:
            job_status = PipelineJobStatus(user_id=user_id, archivo=path_obj.name, tipo=tipo_archivo, estado="procesando")
            db.add(job_status)
        else:
            job_status.estado = "procesando"
            job_status.detalle_error = None
        db.commit()

        total_leidos = 0
        descartados = 0
        registros_guardados = 0
        longitudes_validas = []
        prefijos_counts = {}
        mirna_colapsado_dict = {}

        format_hint = "fastq" if "fastq" in filename_lower or filename_lower.endswith(".fq") else "fasta"

        with open_func(path_obj, "rt") as handle:
            for record in SeqIO.parse(handle, format_hint):
                total_leidos += 1
                mirna_id = record.id.split()[0]
                secuencia = str(record.seq).upper().strip()

                if not secuencia or secuencia.count("N") > 2 or any(c not in "ACGTUN" for c in secuencia):
                    descartados += 1
                    continue

                secuencia = secuencia.replace("N", "")

                if is_mirna:
                    secuencia = limpiar_adaptador_si_existe(secuencia)
                    longitud = len(secuencia)
                    
                    if not (min_len_mirna <= longitud <= max_len_mirna):
                        descartados += 1
                        continue

                    # Colapsado por frecuencia
                    if secuencia in mirna_colapsado_dict:
                        mirna_colapsado_dict[secuencia]["frecuencia"] += 1
                    else:
                        mirna_colapsado_dict[secuencia] = {
                            "miRNA": mirna_id,
                            "frecuencia": 1,
                            "longitud": longitud
                        }
                    
                    registros_guardados += 1

                    if total_leidos >= max_records * 5: 
                        break
                else:
                    longitud = len(secuencia)
                    if longitud < 50:  # Filtro mínimo estructural para mRNA/transcriptoma
                        descartados += 1
                        continue

                    longitudes_validas.append(longitud)
                    prefijo = mirna_id[:6]
                    prefijos_counts[prefijo] = prefijos_counts.get(prefijo, 0) + 1

                    if len(longitudes_validas) >= max_records:
                        break

        if is_mirna and mirna_colapsado_dict:
            batch_data = []
            for seq, info in mirna_colapsado_dict.items():
                batch_data.append({
                    "user_id": user_id,
                    "archivo": path_obj.name,
                    "miRNA": f"{info['miRNA']}_freq_{info['frecuencia']}",
                    "secuencia": seq,
                    "longitud": info["longitud"]
                })
                
                if len(batch_data) >= 5000:
                    db.bulk_insert_mappings(MirnaReport, batch_data)
                    db.commit()
                    batch_data = []

            if batch_data:
                db.bulk_insert_mappings(MirnaReport, batch_data)
                db.commit()
            
            registros_guardados = len(mirna_colapsado_dict)

        if not is_mirna:
            secuencias_validas_count = len(longitudes_validas)
            if secuencias_validas_count == 0:
                raise ValueError(f"El archivo de transcriptoma {path_obj.name} no contiene registros válidos.")

            min_len = min(longitudes_validas)
            max_len = max(longitudes_validas)
            avg_len = sum(longitudes_validas) / secuencias_validas_count
            mediana_len = statistics.median(longitudes_validas)
            top_pref = max(prefijos_counts, key=prefijos_counts.get) if prefijos_counts else "N/A"
            top_pref_count = prefijos_counts[top_pref] if prefijos_counts else 0

            reporte_transcriptoma = TranscriptomeReport(
                user_id=user_id,
                archivo=path_obj.name,
                total_transcritos=total_leidos,
                transcritos_validos=secuencias_validas_count,
                registros_invalidos_descartados=descartados,
                longitud_minima_bp=min_len,
                longitud_maxima_bp=max_len,
                longitud_promedio_bp=avg_len,
                longitud_mediana_bp=mediana_len,
                top_prefijo=top_pref,
                top_prefijo_conteo=top_pref_count
            )
            db.add(reporte_transcriptoma)
            db.commit()
            registros_guardados = secuencias_validas_count

        job_status.estado = "completado"
        job_status.total_leidos = total_leidos
        job_status.validos_guardados = registros_guardados
        job_status.descartados = descartados
        db.commit()

        return {
            "user_id": user_id,
            "archivo": path_obj.name,
            "tipo": tipo_archivo,
            "total_leidos": total_leidos,
            "validos_guardados": registros_guardados,
            "descartados": descartados,
            "estado": "completado"
        }

    except Exception as e:
        db.rollback()
        if 'job_status' in locals() and job_status:
            job_status.estado = "error"
            job_status.detalle_error = str(e)
            db.commit()
        raise ValueError(f"Error procesando el archivo {path_obj.name}: {str(e)}")
    finally:
        db.close()


def check_local_mirbase(referencia_filename: str = "mature.fa") -> Dict[str, Any]:
    ruta_ref = UPLOAD_DIR / referencia_filename
    if ruta_ref.exists():
        return {
            "status": "available",
            "filename": referencia_filename,
            "message": "Referencia local de miRBase encontrada y lista para usar."
        }
    return {
        "status": "missing",
        "filename": referencia_filename,
        "message": "No se encontró el archivo de referencia local en data/Secuencias/."
    }

