import gzip
import time
from pathlib import Path
from typing import Dict, Set, Tuple

from Bio import SeqIO

from app.core.config import MIRBASE_MATURE_FILENAME, SEQUENCES_DIR
from app.models.models import IdentifiedMirna, MirnaInputSummary, PipelineJobStatus
from app.services.database import SessionLocal
from app.services.disk_store import iter_collapsed_reads

# Top-N novel por frecuencia a persistir (el resto queda solo en el TSV colapsado)
MAX_NOVEL_TO_DB = 100


def cargar_referencia_mirbase(
    referencia_filename: str = MIRBASE_MATURE_FILENAME,
) -> Tuple[Dict[str, str], Dict[str, str], Set[str]]:
    """
    Carga mature.fa una vez.
    Retorna:
      - ids_upper → id original
      - seq → id (índice hash para búsqueda O(1))
      - seeds (nt 2–8) → usados solo como señal secundaria
    """
    ruta_ref = SEQUENCES_DIR / referencia_filename
    if not ruta_ref.exists():
        ruta_ref = Path("data") / referencia_filename

    miRBase_ref_ids: Dict[str, str] = {}
    seq_to_id: Dict[str, str] = {}
    miRBase_ref_seeds: Set[str] = set()

    if not ruta_ref.exists():
        # Fallback mínimo para pruebas locales sin mature.fa
        return (
            {"OSA-MIR156A": "osa-MIR156a"},
            {"UGACAGAAGAGAGUGAGCAC": "osa-MIR156a"},
            {"GACAGA"},
        )

    formato = "fasta"
    open_func = gzip.open if str(ruta_ref).endswith(".gz") else open
    with open_func(ruta_ref, "rt") as f_ref:
        for record in SeqIO.parse(f_ref, formato):
            if not record.id or not record.seq:
                continue
            rid = record.id.split()[0]
            seq = str(record.seq).upper().replace("U", "T")
            miRBase_ref_ids[rid.upper()] = rid
            # Preferir primer ID si hay colisiones de secuencia
            seq_to_id.setdefault(seq, rid)
            seq_u = seq.replace("T", "U")
            seq_to_id.setdefault(seq_u, rid)
            if len(seq) >= 8:
                miRBase_ref_seeds.add(seq[1:8])
                miRBase_ref_seeds.add(seq_u[1:8] if len(seq_u) >= 8 else seq[1:8])

    print(f"-> Referencia '{referencia_filename}' indexada: {len(seq_to_id)} secuencias.")
    return miRBase_ref_ids, seq_to_id, miRBase_ref_seeds


def procesar_clasificacion_desde_db_background(
    user_id: int, referencia_filename: str = MIRBASE_MATURE_FILENAME
):
    """
    Lee lecturas colapsadas desde disco (no desde DB), busca en índice hash de mature.fa
    y persiste solo matches + top-N novel.
    """
    db = SessionLocal()
    try:
        print("\n" + "=" * 60)
        print(f"INICIANDO CLASIFICACIÓN miRBase PARA USUARIO ID: {user_id}")
        print("=" * 60)

        t_inicio = time.time()
        _ids, seq_to_id, seeds = cargar_referencia_mirbase(referencia_filename)

        summaries = (
            db.query(MirnaInputSummary)
            .filter(MirnaInputSummary.user_id == user_id)
            .order_by(MirnaInputSummary.id.desc())
            .all()
        )
        if not summaries:
            print(f"No hay resumen de miRNA (collapsed) para el usuario {user_id}.")
            return

        # Usar el resumen más reciente por archivo
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
        novels_buffer = []  # (read_count, payload)

        for summary in latest:
            collapsed = Path(summary.collapsed_path)
            if not collapsed.exists():
                print(f"AVISO: no existe {collapsed}")
                continue

            for seq, read_count, rep_id in iter_collapsed_reads(collapsed):
                seq_norm = seq.upper().replace("U", "T")
                seq_u = seq_norm.replace("T", "U")
                total_procesados += 1

                mirbase_id = seq_to_id.get(seq_norm) or seq_to_id.get(seq_u) or seq_to_id.get(seq.upper())
                if mirbase_id:
                    species = mirbase_id.split("-")[0].lower() if "-" in mirbase_id else None
                    batch.append(
                        {
                            "user_id": user_id,
                            "mirna_id": rep_id or mirbase_id,
                            "mirbase_id": mirbase_id,
                            "mirna_sequence": seq.upper(),
                            "species": species,
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
                    if seed in seeds:
                        batch.append(
                            {
                                "user_id": user_id,
                                "mirna_id": rep_id or f"seed_{seed}",
                                "mirbase_id": None,
                                "mirna_sequence": seq.upper(),
                                "species": None,
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
                                    "mirna_sequence": seq.upper(),
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

        # Solo top-N novel por frecuencia → DB
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
            f"Clasificación OK user={user_id}: {total_procesados:,} secuencias únicas, "
            f"{total_conocidos:,} conocidos/seed, "
            f"{min(len(novels_buffer), MAX_NOVEL_TO_DB)} novel en DB "
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
