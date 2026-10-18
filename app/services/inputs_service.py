import gzip
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from Bio import SeqIO
from fastapi import HTTPException, UploadFile

from app.core.config import (
    ALLOWED_SUFFIXES,
    DEFAULT_MAX_LEN_MIRNA,
    DEFAULT_MIN_LEN_MIRNA,
    IO_CHUNK_BYTES,
    MAX_FILE_SIZE_BYTES,
    MIRBASE_MATURE_FILENAME,
    SEQUENCES_DIR,
    VALIDATION_PEEK_RECORDS,
    user_processed_dir,
    user_raw_dir,
)
from app.models.models import (
    InputFile,
    MirnaInputSummary,
    PipelineJobStatus,
    TranscriptRecord,
    TranscriptomeReport,
    ValidationReport,
)
from app.services.database import SessionLocal
from app.services.disk_store import (
    collapsed_reads_path,
    parse_gene_id_from_header,
    transcript_index_path,
    write_collapsed_reads,
    write_transcript_index,
)

SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)

# Adaptador 3' Illumina Small RNA (TruSeq)
ADAPTADOR_ILLUMINA = "TGGAATTCTCGGGTGCCAAGG"


def limpiar_adaptador_si_existe(secuencia: str) -> str:
    if ADAPTADOR_ILLUMINA in secuencia:
        idx = secuencia.find(ADAPTADOR_ILLUMINA)
        if idx >= 16:
            return secuencia[:idx]
    for i in range(len(ADAPTADOR_ILLUMINA), 6, -1):
        sub = ADAPTADOR_ILLUMINA[:i]
        if secuencia.endswith(sub) and len(secuencia) - i >= 16:
            return secuencia[:-i]
    return secuencia


def _detect_format(filename: str) -> str:
    name = filename.lower()
    for suffix in ALLOWED_SUFFIXES:
        if name.endswith(suffix):
            if "fastq" in suffix or suffix in (".fq", ".fq.gz"):
                return "fastq.gz" if suffix.endswith(".gz") else "fastq"
            return "fasta.gz" if suffix.endswith(".gz") else "fasta"
    return "unknown"


def _assert_allowed_filename(filename: str, tipo_archivo: str) -> str:
    name = (filename or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail=f"Nombre de archivo vacío ({tipo_archivo}).")
    lower = name.lower()
    if not any(lower.endswith(sfx) for sfx in ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato no permitido para '{name}' ({tipo_archivo}). "
                f"Use: FASTA, FASTQ o FASTQ.GZ ({', '.join(ALLOWED_SUFFIXES)})."
            ),
        )
    return name


def _stream_save_upload(file_obj: UploadFile, destino: Path) -> int:
    """Guarda por chunks y cuenta bytes (sin cargar el archivo en RAM)."""
    total = 0
    with open(destino, "wb") as buffer:
        while True:
            chunk = file_obj.file.read(IO_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_SIZE_BYTES:
                buffer.close()
                if destino.exists():
                    destino.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"El archivo '{file_obj.filename}' supera el límite de 1 GB "
                        f"({total / (1024 * 1024):.2f} MB leídos)."
                    ),
                )
            buffer.write(chunk)
    if total == 0:
        if destino.exists():
            destino.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"El archivo '{file_obj.filename}' está vacío.",
        )
    return total


def validar_y_guardar_archivos_obligatorios(
    mirna_file: UploadFile, transcriptome_file: UploadFile, user_id: int
) -> dict:
    """
    Valida entradas obligatorias (formato, no vacío, máx 1 GB c/u),
    guarda en disco por chunks y registra InputFile + ValidationReport.
    """
    if not mirna_file or not mirna_file.filename or not mirna_file.filename.strip():
        raise HTTPException(status_code=400, detail="La entrada de miRNA o small RNA-seq es obligatoria.")
    if not transcriptome_file or not transcriptome_file.filename or not transcriptome_file.filename.strip():
        raise HTTPException(status_code=400, detail="La entrada de transcriptoma (mRNA) es obligatoria.")

    _assert_allowed_filename(mirna_file.filename, "miRNA experimental")
    _assert_allowed_filename(transcriptome_file.filename, "transcriptoma de referencia")

    raw_dir = user_raw_dir(user_id)
    paths_y_tipos = {}
    db = SessionLocal()

    try:
        for file_obj, tipo_archivo, role in (
            (mirna_file, "miRNA experimental", "mirna"),
            (transcriptome_file, "transcriptoma de referencia", "transcriptome"),
        ):
            safe_name = Path(file_obj.filename).name
            ruta_destino = raw_dir / safe_name
            tamano_bytes = _stream_save_upload(file_obj, ruta_destino)
            fmt = _detect_format(safe_name)

            input_row = InputFile(
                user_id=user_id,
                role=role,
                original_filename=safe_name,
                stored_path=str(ruta_destino),
                format=fmt,
                size_bytes=tamano_bytes,
            )
            db.add(input_row)
            db.flush()

            db.add(
                ValidationReport(
                    user_id=user_id,
                    input_file_id=input_row.id,
                    role=role,
                    filename=safe_name,
                    status="procesando",
                    size_ok=True,
                    format_ok=fmt != "unknown",
                    detail="Archivo recibido. Validación estructural en segundo plano...",
                )
            )
            paths_y_tipos[tipo_archivo] = str(ruta_destino)

        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return paths_y_tipos


def _open_seq_file(path_obj: Path):
    is_gz = path_obj.name.lower().endswith(".gz")
    return gzip.open(path_obj, "rt") if is_gz else open(path_obj, "rt")


def _seq_format_hint(filename_lower: str) -> str:
    if "fastq" in filename_lower or filename_lower.endswith(".fq") or filename_lower.endswith(".fq.gz"):
        return "fastq"
    return "fasta"


def peek_validacion_estructura(
    file_path: str,
    role: str,
    user_id: int,
    min_len_mirna: int = DEFAULT_MIN_LEN_MIRNA,
    max_len_mirna: int = DEFAULT_MAX_LEN_MIRNA,
) -> Dict[str, Any]:
    """
    Validación rápida (primeros N registros) para que el GET muestre progreso pronto.
    No escribe colapsados ni transcriptos; solo actualiza ValidationReport.
    """
    path_obj = Path(file_path)
    filename_lower = path_obj.name.lower()
    format_hint = _seq_format_hint(filename_lower)
    format_ok = format_hint in ("fasta", "fastq")

    total = 0
    invalid = 0
    ids_ok = 0
    len_in_range = 0
    quality_sum = 0.0
    quality_n = 0
    structure_ok = False
    detail_parts = []

    try:
        with _open_seq_file(path_obj) as handle:
            for record in SeqIO.parse(handle, format_hint):
                total += 1
                seq = str(record.seq).upper().strip()
                if record.id:
                    ids_ok += 1
                if not seq or any(c not in "ACGTUN" for c in seq):
                    invalid += 1
                else:
                    if role == "mirna":
                        cleaned = limpiar_adaptador_si_existe(seq.replace("N", ""))
                        if min_len_mirna <= len(cleaned) <= max_len_mirna:
                            len_in_range += 1
                    elif len(seq) >= 50:
                        len_in_range += 1

                if format_hint == "fastq" and record.letter_annotations.get("phred_quality"):
                    quals = record.letter_annotations["phred_quality"]
                    if quals:
                        quality_sum += sum(quals) / len(quals)
                        quality_n += 1

                if total >= VALIDATION_PEEK_RECORDS:
                    break
    except Exception as e:
        detail_parts.append(f"Error leyendo estructura: {e}")
        format_ok = False

    if total == 0:
        detail_parts.append("Sin registros parseables en la muestra.")
    else:
        structure_ok = ids_ok > 0 and (total - invalid) > 0
        detail_parts.append(
            f"Muestra: {total} registros, {invalid} inválidos, {len_in_range} con longitud esperada."
        )

    looks_like_mirna = None
    if role == "mirna" and total > 0:
        looks_like_mirna = (len_in_range / total) >= 0.3
        if not looks_like_mirna:
            detail_parts.append(
                f"Pocas lecturas en rango {min_len_mirna}-{max_len_mirna} nt; "
                "verifique que sea small RNA-seq / miRNA."
            )

    avg_q = (quality_sum / quality_n) if quality_n else None
    status = "validando" if structure_ok else "error"

    db = SessionLocal()
    try:
        validation = (
            db.query(ValidationReport)
            .filter_by(user_id=user_id, filename=path_obj.name, role=role)
            .order_by(ValidationReport.id.desc())
            .first()
        )
        if validation:
            validation.status = status if status == "error" else "procesando"
            validation.format_ok = format_ok
            validation.size_ok = True
            validation.structure_ok = structure_ok
            validation.looks_like_mirna = looks_like_mirna
            validation.total_records_sampled = total
            validation.invalid_records = invalid
            validation.avg_quality = avg_q
            validation.detail = " | ".join(detail_parts)
            if status == "error":
                validation.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()

    return {
        "filename": path_obj.name,
        "role": role,
        "format_ok": format_ok,
        "structure_ok": structure_ok,
        "looks_like_mirna": looks_like_mirna,
        "peek_records": total,
        "status": status,
    }


def procesar_y_guardar_muestreo_input(
    file_path: str,
    tipo_archivo: str,
    user_id: int,
    min_len_mirna: int = DEFAULT_MIN_LEN_MIRNA,
    max_len_mirna: int = DEFAULT_MAX_LEN_MIRNA,
) -> Dict[str, Any]:
    """
    Streaming completo (sin truncar): QC + colapsado a disco para miRNA;
    índice de transcritos a disco + metadatos compactos en DB.
    """
    path_obj = Path(file_path)
    filename_lower = path_obj.name.lower()
    is_mirna = "miRNA" in tipo_archivo
    role = "mirna" if is_mirna else "transcriptome"

    db = SessionLocal()
    job_status = None

    try:
        job_status = (
            db.query(PipelineJobStatus)
            .filter_by(user_id=user_id, archivo=path_obj.name, tipo=tipo_archivo)
            .first()
        )
        if not job_status:
            job_status = PipelineJobStatus(
                user_id=user_id, archivo=path_obj.name, tipo=tipo_archivo, estado="procesando"
            )
            db.add(job_status)
        else:
            job_status.estado = "procesando"
            job_status.detalle_error = None
        db.commit()

        validation = (
            db.query(ValidationReport)
            .filter_by(user_id=user_id, filename=path_obj.name, role=role)
            .order_by(ValidationReport.id.desc())
            .first()
        )

        total_leidos = 0
        descartados = 0
        format_hint = _seq_format_hint(filename_lower)
        format_ok = format_hint in ("fasta", "fastq")
        structure_ok = True
        quality_sum = 0.0
        quality_n = 0
        looks_like_mirna = None

        if is_mirna:
            mirna_colapsado: Dict[str, Dict] = {}
            lengths = []

            with _open_seq_file(path_obj) as handle:
                for record in SeqIO.parse(handle, format_hint):
                    total_leidos += 1
                    mirna_id = record.id.split()[0]
                    secuencia = str(record.seq).upper().strip()

                    if format_hint == "fastq" and record.letter_annotations.get("phred_quality"):
                        quals = record.letter_annotations["phred_quality"]
                        if quals:
                            quality_sum += sum(quals) / len(quals)
                            quality_n += 1

                    if not secuencia or secuencia.count("N") > 2 or any(c not in "ACGTUN" for c in secuencia):
                        descartados += 1
                        continue

                    secuencia = secuencia.replace("N", "")
                    secuencia = limpiar_adaptador_si_existe(secuencia)
                    longitud = len(secuencia)

                    if not (min_len_mirna <= longitud <= max_len_mirna):
                        descartados += 1
                        continue

                    lengths.append(longitud)
                    if secuencia in mirna_colapsado:
                        mirna_colapsado[secuencia]["read_count"] += 1
                    else:
                        mirna_colapsado[secuencia] = {
                            "read_count": 1,
                            "representative_id": mirna_id,
                        }

            collapsed_path = collapsed_reads_path(user_id, path_obj.name)
            unique_n = write_collapsed_reads(collapsed_path, mirna_colapsado)
            reads_after_qc = sum(v["read_count"] for v in mirna_colapsado.values())
            looks_like_mirna = unique_n > 0 and (
                statistics.median(lengths) <= max_len_mirna if lengths else False
            )

            # Reemplazar resumen previo del mismo archivo
            db.query(MirnaInputSummary).filter_by(user_id=user_id, archivo=path_obj.name).delete()
            summary = MirnaInputSummary(
                user_id=user_id,
                archivo=path_obj.name,
                collapsed_path=str(collapsed_path),
                total_reads=total_leidos,
                reads_after_qc=reads_after_qc,
                unique_sequences=unique_n,
                discarded=descartados,
                min_len=min(lengths) if lengths else None,
                max_len=max(lengths) if lengths else None,
            )
            db.add(summary)
            registros_guardados = unique_n
            structure_ok = unique_n > 0

        else:
            longitudes_validas = []
            prefijos_counts: Dict[str, int] = {}
            index_rows = []
            user_processed_dir(user_id)

            with _open_seq_file(path_obj) as handle:
                for record in SeqIO.parse(handle, format_hint):
                    total_leidos += 1
                    tid = record.id.split()[0]
                    secuencia = str(record.seq).upper().strip()

                    if not secuencia or any(c not in "ACGTUN" for c in secuencia):
                        descartados += 1
                        continue

                    longitud = len(secuencia)
                    if longitud < 50:
                        descartados += 1
                        continue

                    longitudes_validas.append(longitud)
                    prefijo = tid[:6]
                    prefijos_counts[prefijo] = prefijos_counts.get(prefijo, 0) + 1
                    gene_id = parse_gene_id_from_header(record.description)
                    index_rows.append(
                        {
                            "transcript_id": tid,
                            "gene_id": gene_id,
                            "length_bp": longitud,
                            "file_offset": None,
                            "annotation": (record.description or "")[:500],
                        }
                    )

            if not longitudes_validas:
                raise ValueError(f"El archivo de transcriptoma {path_obj.name} no contiene registros válidos.")

            idx_path = transcript_index_path(user_id, path_obj.name)
            write_transcript_index(idx_path, index_rows)

            top_pref = max(prefijos_counts, key=prefijos_counts.get) if prefijos_counts else "N/A"
            top_pref_count = prefijos_counts.get(top_pref, 0)

            # Limpiar reportes previos del mismo archivo
            prev = (
                db.query(TranscriptomeReport)
                .filter_by(user_id=user_id, archivo=path_obj.name)
                .all()
            )
            for p in prev:
                db.query(TranscriptRecord).filter_by(transcriptome_id=p.id).delete()
                db.delete(p)
            db.flush()

            reporte = TranscriptomeReport(
                user_id=user_id,
                archivo=path_obj.name,
                sequences_path=str(path_obj),
                index_path=str(idx_path),
                total_transcritos=total_leidos,
                transcritos_validos=len(longitudes_validas),
                registros_invalidos_descartados=descartados,
                longitud_minima_bp=min(longitudes_validas),
                longitud_maxima_bp=max(longitudes_validas),
                longitud_promedio_bp=sum(longitudes_validas) / len(longitudes_validas),
                longitud_mediana_bp=statistics.median(longitudes_validas),
                top_prefijo=top_pref,
                top_prefijo_conteo=top_pref_count,
            )
            db.add(reporte)
            db.flush()

            # Metadatos compactos en DB (sin secuencia)
            batch = []
            for row in index_rows:
                batch.append(
                    {
                        "user_id": user_id,
                        "transcriptome_id": reporte.id,
                        "transcript_id": row["transcript_id"],
                        "gene_id": row["gene_id"],
                        "length_bp": row["length_bp"],
                        "file_offset": row["file_offset"],
                        "annotation": row["annotation"],
                    }
                )
                if len(batch) >= 2000:
                    db.bulk_insert_mappings(TranscriptRecord, batch)
                    db.commit()
                    batch = []
            if batch:
                db.bulk_insert_mappings(TranscriptRecord, batch)
                db.commit()

            registros_guardados = len(longitudes_validas)
            structure_ok = True

        if validation:
            validation.status = "completado"
            validation.format_ok = format_ok
            validation.size_ok = True
            validation.structure_ok = structure_ok
            validation.looks_like_mirna = looks_like_mirna
            validation.total_records_sampled = total_leidos
            validation.invalid_records = descartados
            validation.avg_quality = (quality_sum / quality_n) if quality_n else None
            warn = ""
            if role == "mirna" and looks_like_mirna is False:
                warn = " Advertencia: el perfil de longitudes no parece miRNA/sRNA típico."
            validation.detail = (
                f"QC completo. Leídos={total_leidos}, descartados={descartados}, "
                f"persistidos(metadatos)={registros_guardados}.{warn}"
            )
            validation.completed_at = datetime.now(timezone.utc)

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
            "estado": "completado",
            "storage": "disk_sequences_db_metadata",
        }

    except Exception as e:
        db.rollback()
        if job_status:
            job_status.estado = "error"
            job_status.detalle_error = str(e)
            db.commit()
        if "validation" in locals() and validation:
            validation.status = "error"
            validation.detail = str(e)
            db.commit()
        raise ValueError(f"Error procesando el archivo {path_obj.name}: {str(e)}")
    finally:
        db.close()


def check_local_mirbase(referencia_filename: str = MIRBASE_MATURE_FILENAME) -> Dict[str, Any]:
    ruta_ref = SEQUENCES_DIR / referencia_filename
    if ruta_ref.exists():
        return {
            "status": "available",
            "filename": referencia_filename,
            "message": "Referencia local de miRBase encontrada y lista para usar.",
        }
    return {
        "status": "missing",
        "filename": referencia_filename,
        "message": "No se encontró el archivo de referencia local en data/Secuencias/.",
    }
