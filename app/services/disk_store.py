"""
Persistencia en disco de secuencias/colapsados/índices.
La DB solo guarda rutas y metadatos.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

from app.core.config import user_processed_dir


COLLAPSED_HEADER = ["sequence", "read_count", "representative_id"]
TRANSCRIPT_INDEX_HEADER = ["transcript_id", "gene_id", "length_bp", "file_offset", "annotation"]


def collapsed_reads_path(user_id: int, source_filename: str) -> Path:
    safe = Path(source_filename).stem.replace(" ", "_")
    return user_processed_dir(user_id) / f"collapsed_{safe}.tsv"


def transcript_index_path(user_id: int, source_filename: str) -> Path:
    safe = Path(source_filename).stem.replace(" ", "_")
    return user_processed_dir(user_id) / f"transcript_index_{safe}.tsv"


def write_collapsed_reads(
    path: Path,
    collapsed: Dict[str, Dict],
) -> int:
    """
    Escribe seq → {read_count, representative_id}.
    Retorna número de secuencias únicas.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLLAPSED_HEADER, delimiter="\t")
        writer.writeheader()
        for seq, info in collapsed.items():
            writer.writerow(
                {
                    "sequence": seq,
                    "read_count": info["read_count"],
                    "representative_id": info.get("representative_id", ""),
                }
            )
    return len(collapsed)


def iter_collapsed_reads(path: Path) -> Iterator[Tuple[str, int, str]]:
    """Streaming: (sequence, read_count, representative_id)."""
    if not path.exists():
        yield from ()
        return
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            yield (
                row["sequence"],
                int(row["read_count"]),
                row.get("representative_id") or "",
            )


def write_transcript_index(
    path: Path,
    rows: Iterable[dict],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRANSCRIPT_INDEX_HEADER, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "transcript_id": row["transcript_id"],
                    "gene_id": row.get("gene_id") or "",
                    "length_bp": row["length_bp"],
                    "file_offset": row.get("file_offset") if row.get("file_offset") is not None else "",
                    "annotation": row.get("annotation") or "",
                }
            )
            count += 1
    return count


def parse_gene_id_from_header(header: str) -> Optional[str]:
    """Heurística simple: gene=XXX o segundo token del header FASTA."""
    if not header:
        return None
    parts = header.replace("|", " ").split()
    for p in parts:
        if p.lower().startswith("gene=") or p.lower().startswith("gene:"):
            return p.split("=", 1)[-1].split(":", 1)[-1]
    if len(parts) >= 2 and not parts[1].startswith("("):
        return parts[1]
    return None
