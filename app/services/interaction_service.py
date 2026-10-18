import gzip
from pathlib import Path
from typing import List, Optional

import pandas as pd
from Bio import SeqIO

from app.core.config import LABEL_SOURCE_SYNTHETIC_NEGATIVE, LABEL_SOURCE_UNLABELED, user_raw_dir
from app.models.models import (
    IdentifiedMirna,
    InteractionCandidate,
    PipelineJobStatus,
    TranscriptomeReport,
)
from app.services.database import SessionLocal


def _open_seq_file(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def _resolve_transcriptome_path(user_id: int, nombre_archivo: str, sequences_path: Optional[str]) -> Path:
    if sequences_path and Path(sequences_path).exists():
        return Path(sequences_path)
    candidate = user_raw_dir(user_id) / nombre_archivo
    if candidate.exists():
        return candidate
    fallback = Path("data/Secuencias") / nombre_archivo
    return fallback


def calcular_caracteristicas_streaming(
    lista_mirnas: list,
    transcriptome_path: Path,
    max_pares_por_mirna: int = 5,
) -> pd.DataFrame:
    """
    miRNAs en memoria (pocos: solo identificados).
    Transcriptoma en streaming — no carga el catálogo completo en un dict.
    """
    # Precomputar seeds
    mirna_ready = []
    for item in lista_mirnas:
        seq = (item.get("secuencia") or "").upper().replace("U", "T")
        if len(seq) < 8:
            continue
        mirna_ready.append(
            {
                "miRNA": item["miRNA"],
                "secuencia": seq,
                "seed": seq[1:8],
                "read_count": item.get("read_count", 1),
                "match_type": item.get("match_type", ""),
            }
        )

    pares_por_mirna = {m["miRNA"]: 0 for m in mirna_ready}
    datos = []

    if not transcriptome_path.exists():
        raise ValueError(f"No se encontró transcriptoma en {transcriptome_path}")

    fmt = "fastq" if "fastq" in transcriptome_path.name.lower() else "fasta"
    with _open_seq_file(transcriptome_path) as handle:
        for record in SeqIO.parse(handle, fmt):
            t_id = record.id.split()[0]
            t_seq = str(record.seq).upper().replace("U", "T")
            if len(t_seq) < 50:
                continue

            for m in mirna_ready:
                if pares_por_mirna[m["miRNA"]] >= max_pares_por_mirna:
                    continue
                seed = m["seed"]
                pos = t_seq.find(seed)
                if pos < 0:
                    continue

                coincidencias = t_seq.count(seed)
                complementariedad = round(min(0.98, 0.75 + (0.04 * min(coincidencias, 5))), 2)
                mismatches = max(0, 2 - min(coincidencias, 2))
                # ΔG / accesibilidad: placeholders documentados (nullables en DB);
                # se rellenan con heurística débil hasta integrar ViennaRNA u otra herramienta.
                delta_g = round(-12.5 - (coincidencias * 3.2), 2)
                accesibilidad = round(0.50 + ((len(t_id) % 45) / 100.0), 2)
                alignment_score = round(
                    (complementariedad * 0.4) + ((abs(delta_g) / 50.0) * 0.4) + (accesibilidad * 0.2),
                    3,
                )

                datos.append(
                    {
                        "mirna_id": m["miRNA"],
                        "mrna_id": t_id,
                        "gene_id": None,
                        "seed_match": 1,
                        "seed_position": pos,
                        "match_length": len(seed),
                        "num_matches": coincidencias,
                        "mismatches": mismatches,
                        "gaps": 0,
                        "indels": 0,
                        "complementarity": complementariedad,
                        "alignment_score": alignment_score,
                        "site_position": pos,
                        "site_length": len(seed),
                        "delta_g": delta_g,
                        "accessibility": accesibilidad,
                        "conservation": None,
                        "mirna_expression": float(m["read_count"]),
                        "mrna_expression": None,
                        "label": None,
                        "label_source": LABEL_SOURCE_UNLABELED,
                    }
                )
                pares_por_mirna[m["miRNA"]] += 1

            if all(v >= max_pares_por_mirna for v in pares_por_mirna.values()):
                break

    # Negativas sintéticas: mismos miRNA, primer transcrito sin seed (muestreo simple)
    # Se generan en una segunda pasada corta si hay candidatos positivos.
    if datos:
        datos.extend(_synthetic_negatives(mirna_ready, transcriptome_path, n_per_mirna=2))

    return pd.DataFrame(datos)


def _synthetic_negatives(mirna_ready: list, transcriptome_path: Path, n_per_mirna: int = 2) -> List[dict]:
    """Pares sin complementaridad de seed → label=0, label_source=synthetic_negative."""
    fmt = "fastq" if "fastq" in transcriptome_path.name.lower() else "fasta"
    out = []
    counts = {m["miRNA"]: 0 for m in mirna_ready}

    with _open_seq_file(transcriptome_path) as handle:
        for record in SeqIO.parse(handle, fmt):
            t_id = record.id.split()[0]
            t_seq = str(record.seq).upper().replace("U", "T")
            if len(t_seq) < 50:
                continue
            for m in mirna_ready:
                if counts[m["miRNA"]] >= n_per_mirna:
                    continue
                if m["seed"] in t_seq:
                    continue
                out.append(
                    {
                        "mirna_id": m["miRNA"],
                        "mrna_id": t_id,
                        "gene_id": None,
                        "seed_match": 0,
                        "seed_position": None,
                        "match_length": 0,
                        "num_matches": 0,
                        "mismatches": len(m["seed"]),
                        "gaps": 0,
                        "indels": 0,
                        "complementarity": 0.1,
                        "alignment_score": 0.05,
                        "site_position": None,
                        "site_length": 0,
                        "delta_g": 0.0,
                        "accessibility": 0.3,
                        "conservation": None,
                        "mirna_expression": float(m["read_count"]),
                        "mrna_expression": None,
                        "label": 0,
                        "label_source": LABEL_SOURCE_SYNTHETIC_NEGATIVE,
                    }
                )
                counts[m["miRNA"]] += 1
            if all(v >= n_per_mirna for v in counts.values()):
                break
    return out


def procesar_interacciones_background(user_id: int, transcriptome_id: int, max_pares_por_mirna: int = 5):
    db = SessionLocal()
    job_nombre = f"Prediction Interaction - Run #{transcriptome_id}"
    job_status = None

    try:
        print(f"\nCálculo de interacciones user={user_id} transcriptome={transcriptome_id}")

        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if not job_status:
            job_status = PipelineJobStatus(
                user_id=user_id,
                archivo=f"transcriptome_{transcriptome_id}",
                tipo=job_nombre,
                estado="procesando",
            )
            db.add(job_status)
        else:
            job_status.estado = "procesando"
            job_status.detalle_error = None
        db.commit()

        transcriptome_record = (
            db.query(TranscriptomeReport).filter_by(user_id=user_id, id=transcriptome_id).first()
        )
        if not transcriptome_record:
            raise ValueError(f"No se encontró transcriptoma {transcriptome_id} para user {user_id}.")

        # Preferir conocidos/seed; incluir novel top-N ya persistidos
        mirnas_guardados = db.query(IdentifiedMirna).filter_by(user_id=user_id).all()
        if not mirnas_guardados:
            raise ValueError(f"No hay miRNAs identificados para el usuario {user_id}.")

        lista_mirnas = [
            {
                "miRNA": m.mirbase_id or m.mirna_id,
                "secuencia": m.mirna_sequence,
                "read_count": m.read_count,
                "match_type": m.match_type,
            }
            for m in mirnas_guardados
        ]

        t_path = _resolve_transcriptome_path(
            user_id, transcriptome_record.archivo, transcriptome_record.sequences_path
        )
        df = calcular_caracteristicas_streaming(lista_mirnas, t_path, max_pares_por_mirna)

        db.query(InteractionCandidate).filter_by(
            user_id=user_id, transcriptome_id=transcriptome_id
        ).delete()
        db.commit()

        batch = []
        BATCH_SIZE = 2000
        for _, row in df.iterrows():
            batch.append(
                {
                    "user_id": user_id,
                    "transcriptome_id": transcriptome_id,
                    "mirna_id": row["mirna_id"],
                    "mrna_id": row["mrna_id"],
                    "gene_id": row.get("gene_id"),
                    "seed_match": int(row["seed_match"]),
                    "seed_position": row.get("seed_position"),
                    "match_length": row.get("match_length"),
                    "num_matches": row.get("num_matches"),
                    "mismatches": int(row["mismatches"]),
                    "gaps": int(row.get("gaps") or 0),
                    "indels": int(row.get("indels") or 0),
                    "complementarity": float(row["complementarity"]),
                    "alignment_score": float(row["alignment_score"]) if row.get("alignment_score") is not None else None,
                    "site_position": row.get("site_position"),
                    "site_length": row.get("site_length"),
                    "delta_g": float(row["delta_g"]) if row.get("delta_g") is not None else None,
                    "accessibility": float(row["accessibility"]) if row.get("accessibility") is not None else None,
                    "conservation": row.get("conservation"),
                    "mirna_expression": row.get("mirna_expression"),
                    "mrna_expression": row.get("mrna_expression"),
                    "label": int(row["label"]) if row.get("label") is not None else None,
                    "label_source": row.get("label_source") or LABEL_SOURCE_UNLABELED,
                }
            )
            if len(batch) >= BATCH_SIZE:
                db.bulk_insert_mappings(InteractionCandidate, batch)
                db.commit()
                batch = []
        if batch:
            db.bulk_insert_mappings(InteractionCandidate, batch)
            db.commit()

        job_status.estado = "completado"
        job_status.validos_guardados = len(df)
        db.commit()
        print(f"Interacciones guardadas: {len(df)} (user={user_id})")

    except Exception as e:
        db.rollback()
        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if job_status:
            job_status.estado = "error"
            job_status.detalle_error = str(e)
            db.commit()
        print(f"Error interacciones user={user_id}: {e}")
    finally:
        db.close()
