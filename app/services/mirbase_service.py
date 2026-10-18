import gzip
import re
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from Bio import SeqIO

from app.core.config import MIRBASE_MATURE_FILENAME, SEQUENCES_DIR
from app.models.models import IdentifiedMirna, MirnaInputSummary, PipelineJobStatus
from app.services.database import SessionLocal
from app.services.disk_store import iter_collapsed_reads

# Top-N novel por frecuencia a persistir (el resto queda solo en el TSV colapsado)
MAX_NOVEL_TO_DB = 100

# Entrada de índice: mirbase_id, mimat, species, sequence
RefEntry = Dict[str, str]


def parse_mirbase_header(description: str) -> Tuple[str, Optional[str], str]:
    """
    Parsea cabecera miRBase, p.ej.:
      cel-let-7-5p MIMAT0000001 Caenorhabditis elegans let-7-5p
      osa-miR156a MIMAT0000645 Oryza sativa

    Retorna: (mirbase_id, mimat_id|None, species_nombre_completo)
    """
    parts = (description or "").strip().split()
    if not parts:
        return "unknown", None, "unknown"

    mirbase_id = parts[0]
    idx = 1
    mimat = None
    if idx < len(parts) and parts[idx].upper().startswith("MIMAT"):
        mimat = parts[idx]
        idx += 1

    remaining = parts[idx:]
    abbrev = mirbase_id.split("-")[0] if "-" in mirbase_id else mirbase_id

    if not remaining:
        return mirbase_id, mimat, abbrev

    # Sufijo del producto (let-7-5p, miR156a, …) suele ir al final
    product = mirbase_id.split("-", 1)[-1] if "-" in mirbase_id else ""
    if len(remaining) >= 2 and (
        remaining[-1] == product
        or remaining[-1] in mirbase_id
        or re.fullmatch(r"(miR|MIR|let|lin|lsy)[\w.-]*", remaining[-1], re.I)
    ):
        species = " ".join(remaining[:-1])
    elif len(remaining) >= 2:
        # Binomio científico por defecto (Genus species); si hay más palabras sin producto, úsalas
        # p.ej. "Caenorhabditis elegans" o nombres más largos sin token de producto
        species = " ".join(remaining)
        # Si el último token parece ID corto (tiene dígitos y guiones), quitarlo
        if re.search(r"\d", remaining[-1]) and "-" in remaining[-1]:
            species = " ".join(remaining[:-1])
    else:
        species = remaining[0]

    return mirbase_id, mimat, species.strip() or abbrev


def cargar_referencia_mirbase(
    referencia_filename: str = MIRBASE_MATURE_FILENAME,
) -> Tuple[Dict[str, RefEntry], Dict[str, RefEntry]]:
    """
    Carga mature.fa una vez.
    Retorna:
      - seq_index: secuencia (T y U) → {mirbase_id, mimat, species}
      - seed_index: seed nt 2–8 → mismo dict (primer hit)
    """
    ruta_ref = SEQUENCES_DIR / referencia_filename
    if not ruta_ref.exists():
        ruta_ref = Path("data") / referencia_filename

    seq_index: Dict[str, RefEntry] = {}
    seed_index: Dict[str, RefEntry] = {}

    if not ruta_ref.exists():
        demo = {
            "mirbase_id": "osa-miR156a",
            "mimat": "MIMAT0000645",
            "species": "Oryza sativa",
        }
        return (
            {"UGACAGAAGAGAGUGAGCAC": demo, "TGACAGAAGAGAGTGAGCAC": demo},
            {"GACAGAAGAG": demo, "GACAGAA": demo},
        )

    open_func = gzip.open if str(ruta_ref).endswith(".gz") else open
    with open_func(ruta_ref, "rt", encoding="utf-8", errors="replace") as f_ref:
        for record in SeqIO.parse(f_ref, "fasta"):
            if not record.seq:
                continue
            desc = record.description or record.id or ""
            mirbase_id, mimat, species = parse_mirbase_header(desc)
            entry: RefEntry = {
                "mirbase_id": mirbase_id,
                "mimat": mimat or "",
                "species": species,
            }
            seq_raw = str(record.seq).upper()
            seq_t = seq_raw.replace("U", "T")
            seq_u = seq_t.replace("T", "U")

            seq_index.setdefault(seq_t, entry)
            seq_index.setdefault(seq_u, entry)
            seq_index.setdefault(seq_raw, entry)

            if len(seq_t) >= 8:
                seed_t = seq_t[1:8]
                seed_u = seq_u[1:8]
                seed_index.setdefault(seed_t, entry)
                seed_index.setdefault(seed_u, entry)

    print(
        f"-> miRBase '{ruta_ref}' indexada: {len(seq_index)//2} secuencias approx, "
        f"{len(seed_index)} seeds."
    )
    return seq_index, seed_index


def procesar_clasificacion_desde_db_background(
    user_id: int, referencia_filename: str = MIRBASE_MATURE_FILENAME
):
    """
    Lee lecturas colapsadas desde disco, busca en índice hash de mature.fa
    y persiste matches con mirbase_id + species correctos (+ top-N novel).
    """
    db = SessionLocal()
    try:
        print("\n" + "=" * 60)
        print(f"INICIANDO CLASIFICACIÓN miRBase PARA USUARIO ID: {user_id}")
        print("=" * 60)

        t_inicio = time.time()
        seq_index, seed_index = cargar_referencia_mirbase(referencia_filename)

        summaries = (
            db.query(MirnaInputSummary)
            .filter(MirnaInputSummary.user_id == user_id)
            .order_by(MirnaInputSummary.id.desc())
            .all()
        )
        if not summaries:
            print(f"No hay resumen de miRNA (collapsed) para el usuario {user_id}.")
            db.query(PipelineJobStatus).filter(
                PipelineJobStatus.user_id == user_id,
                PipelineJobStatus.tipo == "miRBase Classification",
            ).update({"estado": "error", "detalle_error": "Sin MirnaInputSummary"})
            db.commit()
            return

        seen_files = set()
        latest = []
        for s in summaries:
            if s.archivo in seen_files:
                continue
            seen_files.add(s.archivo)
            latest.append(s)

        db.query(IdentifiedMirna).filter(IdentifiedMirna.user_id == user_id).delete()
        db.commit()

        batch = []
        BATCH_SIZE = 2000
        total_procesados = 0
        total_conocidos = 0
        novels_buffer = []

        for summary in latest:
            collapsed = Path(summary.collapsed_path)
            if not collapsed.exists():
                print(f"AVISO: no existe {collapsed}")
                continue

            for seq, read_count, rep_id in iter_collapsed_reads(collapsed):
                seq_norm = seq.upper().replace("U", "T")
                seq_u = seq_norm.replace("T", "U")
                total_procesados += 1

                entry = (
                    seq_index.get(seq_norm)
                    or seq_index.get(seq_u)
                    or seq_index.get(seq.upper())
                )
                if entry:
                    batch.append(
                        {
                            "user_id": user_id,
                            "mirna_id": rep_id or entry["mirbase_id"],
                            "mirbase_id": entry["mirbase_id"],
                            "mirna_sequence": seq.upper().replace("T", "U"),
                            "species": entry["species"],
                            "read_count": read_count,
                            "match_type": "exact",
                            "mismatches": 0,
                            "alignment_score": 1.0,
                            "source_file": summary.archivo,
                        }
                    )
                    total_conocidos += 1
                else:
                    seed = seq_norm[1:8] if len(seq_norm) >= 8 else seq_norm
                    seed_entry = seed_index.get(seed) or seed_index.get(seed.replace("T", "U"))
                    if seed_entry:
                        batch.append(
                            {
                                "user_id": user_id,
                                "mirna_id": rep_id or seed_entry["mirbase_id"],
                                "mirbase_id": seed_entry["mirbase_id"],
                                "mirna_sequence": seq.upper().replace("T", "U"),
                                "species": seed_entry["species"],
                                "read_count": read_count,
                                "match_type": "seed",
                                "mismatches": 0,
                                "alignment_score": 0.7,
                                "source_file": summary.archivo,
                            }
                        )
                        total_conocidos += 1
                    else:
                        novels_buffer.append(
                            (
                                read_count,
                                {
                                    "user_id": user_id,
                                    "mirna_id": rep_id or f"novel_{total_procesados}",
                                    "mirbase_id": None,
                                    "mirna_sequence": seq.upper().replace("T", "U"),
                                    "species": None,
                                    "read_count": read_count,
                                    "match_type": "novel",
                                    "mismatches": 0,
                                    "alignment_score": 0.0,
                                    "source_file": summary.archivo,
                                },
                            )
                        )

                if len(batch) >= BATCH_SIZE:
                    db.bulk_insert_mappings(IdentifiedMirna, batch)
                    db.commit()
                    batch = []

        novels_buffer.sort(key=lambda x: x[0], reverse=True)
        for _, payload in novels_buffer[:MAX_NOVEL_TO_DB]:
            batch.append(payload)
            if len(batch) >= BATCH_SIZE:
                db.bulk_insert_mappings(IdentifiedMirna, batch)
                db.commit()
                batch = []

        if batch:
            db.bulk_insert_mappings(IdentifiedMirna, batch)
            db.commit()

        db.query(PipelineJobStatus).filter(
            PipelineJobStatus.user_id == user_id,
            PipelineJobStatus.tipo == "miRBase Classification",
        ).update({"estado": "completado", "validos_guardados": total_conocidos})
        db.commit()

        t_total = time.time() - t_inicio
        print(
            f"Clasificación OK user={user_id}: {total_procesados:,} únicas, "
            f"{total_conocidos:,} con mirbase_id/species, "
            f"{min(len(novels_buffer), MAX_NOVEL_TO_DB)} novel "
            f"({t_total:.2f}s)"
        )

    except Exception as e:
        db.rollback()
        db.query(PipelineJobStatus).filter(
            PipelineJobStatus.user_id == user_id,
            PipelineJobStatus.tipo == "miRBase Classification",
        ).update({"estado": "error", "detalle_error": str(e)})
        db.commit()
        print(f"Error clasificación user={user_id}: {e}")
    finally:
        db.close()
