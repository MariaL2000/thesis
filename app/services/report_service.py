"""Informe PDF de la búsqueda actual de un usuario (Matplotlib PdfPages)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from app.core.config import EXPORTS_DIR, VISUALS_DIR
from app.models.models import IdentifiedMirna, InteractionCandidate, MLModelRun, TranscriptomeReport, User
from app.services.binding_affinity import describe_binding_frame
from app.services.database import SessionLocal

def _finite(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


PLOT_FILES = [
    "curva_roc_comparativa.png",
    "curva_precision_recall.png",
    "importancia_variables.png",
    "complementariedad_hist.png",
    "delta_g_vs_complementariedad.png",
    "seed_match_conteo.png",
]


def _latest_context(db, user_id: int):
    user = db.query(User).filter_by(id=user_id).first()
    if not user:
        raise ValueError(f"Usuario {user_id} no encontrado.")
    transcriptome = (
        db.query(TranscriptomeReport)
        .filter_by(user_id=user_id)
        .order_by(TranscriptomeReport.id.desc())
        .first()
    )
    if not transcriptome:
        raise ValueError("Este usuario todavía no tiene una búsqueda con transcriptoma.")
    rows = (
        db.query(InteractionCandidate)
        .filter_by(user_id=user_id, transcriptome_id=transcriptome.id)
        .all()
    )
    mirnas = db.query(IdentifiedMirna).filter_by(user_id=user_id).all()
    runs = (
        db.query(MLModelRun)
        .filter_by(user_id=user_id, transcriptome_id=transcriptome.id)
        .order_by(MLModelRun.trained_at.desc())
        .all()
    )
    return user, transcriptome, rows, mirnas, runs


def report_document(user_id: int) -> dict:
    db = SessionLocal()
    try:
        _user, transcriptome, rows, mirnas, runs = _latest_context(db, user_id)
        df = pd.DataFrame(
            [
                {
                    "mirna_id": r.mirna_id,
                    "mrna_id": r.mrna_id,
                    "complementarity": r.complementarity,
                    "delta_g": r.delta_g,
                    "seed_match": r.seed_match,
                    "mismatches": r.mismatches,
                    "accessibility": r.accessibility,
                    "alignment_score": r.alignment_score,
                    "ml_probability": r.ml_probability,
                    "label": r.label,
                    "label_source": r.label_source,
                }
                for r in rows
            ]
        )
        stats = describe_binding_frame(df) if len(df) else {}
        plots_dir = VISUALS_DIR / f"user_{user_id}_run_{transcriptome.id}"
        plots = [name for name in PLOT_FILES if (plots_dir / name).exists()]
        latest_version = runs[0].model_version if runs else None
        top = []
        if len(df) and "ml_probability" in df.columns:
            ordered = df.sort_values("ml_probability", ascending=False, na_position="last").head(8)
            top = json.loads(ordered.to_json(orient="records"))

        return {
            "user_id": user_id,
            "transcriptome_id": transcriptome.id,
            "archivo": transcriptome.archivo,
            "generated_at": datetime.now(timezone.utc),
            "n_interactions": len(rows),
            "n_identified_mirnas": len(mirnas),
            "mean_complementarity": _finite(stats.get("mean_complementarity")),
            "mean_delta_g": _finite(stats.get("mean_delta_g")),
            "seed_match_fraction": _finite(stats.get("seed_match_fraction")),
            "model_version": latest_version,
            "models": [
                {
                    "model_name": r.model_name,
                    "model_version": r.model_version,
                    "accuracy": r.accuracy,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1_score": r.f1_score,
                    "roc_auc": r.roc_auc,
                }
                for r in runs
                if latest_version is None or r.model_version == latest_version
            ],
            "plots": plots,
            "pdf_url": f"/api/v1/report/{user_id}/pdf",
            "top_pairs": top,
        }
    finally:
        db.close()


def _text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.95, title, fontsize=16, fontweight="bold", va="top")
    y = 0.90
    for line in lines:
        fig.text(0.08, y, line, fontsize=10, va="top", wrap=True)
        y -= 0.028
        if y < 0.06:
            break
    pdf.savefig(fig)
    plt.close(fig)


def _table_page(pdf: PdfPages, title: str, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, pad=12)
    if frame.empty:
        ax.text(0.0, 0.5, "Sin filas para esta tabla.")
    else:
        shown = frame.head(12).copy()
        for col in shown.columns:
            shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)[:28]))
        table = ax.table(cellText=shown.values, colLabels=list(shown.columns), loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
    pdf.savefig(fig)
    plt.close(fig)


def write_user_pdf(user_id: int) -> Path:
    doc = report_document(user_id)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = EXPORTS_DIR / f"user_{user_id}_transcriptome_{doc['transcriptome_id']}.pdf"
    plots_dir = VISUALS_DIR / f"user_{user_id}_run_{doc['transcriptome_id']}"

    lines = [
        f"Usuario: {user_id}    Transcriptoma: {doc['archivo']} (id {doc['transcriptome_id']})",
        f"Generado: {doc['generated_at'].strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Metodología",
        "1. QC de small RNA y colapsado de lecturas idénticas.",
        "2. Identificación contra mature.fa (índice por secuencia).",
        "3. Sitios en el mRNA por complemento reverso del seed (Biopython).",
        "4. Complementariedad Watson-Crick / wobble G-U y ΔG por vecinos próximos (NumPy).",
        "5. Accesibilidad aproximada: fracción A/U en la ventana del sitio.",
        "6. Modelos: Regresión logística (relación lineal interpretable) y Random Forest",
        "   (relaciones no lineales e importancia de variables). Se reentrenan con el",
        "   historial etiquetado del mismo usuario.",
        "",
        f"miRNA identificados: {doc['n_identified_mirnas']}",
        f"Pares de esta búsqueda: {doc['n_interactions']}",
        f"Complementariedad media: {doc['mean_complementarity']}",
        f"ΔG medio (kcal/mol): {doc['mean_delta_g']}",
        f"Fracción con seed match: {doc['seed_match_fraction']}",
        f"Versión del modelo: {doc['model_version']}",
    ]
    for model in doc["models"]:
        lines.append(
            f"- {model['model_name']}: acc={model['accuracy']} prec={model['precision']} "
            f"rec={model['recall']} f1={model['f1_score']} auc={model['roc_auc']}"
        )

    with PdfPages(out) as pdf:
        _text_page(pdf, "Informe de interacción miRNA–mRNA", lines)
        top = pd.DataFrame(doc["top_pairs"])
        keep = [c for c in ("mirna_id", "mrna_id", "complementarity", "delta_g", "seed_match", "ml_probability", "label_source") if c in top.columns]
        _table_page(pdf, "Pares con mayor probabilidad en esta búsqueda", top[keep] if keep else top)
        for name in doc["plots"]:
            image_path = plots_dir / name
            if not image_path.exists():
                continue
            img = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(11.69, 8.27))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(name.replace("_", " ").replace(".png", ""), loc="left")
            pdf.savefig(fig)
            plt.close(fig)
        d = pdf.infodict()
        d["Title"] = f"Informe miRNA usuario {user_id}"
        d["CreationDate"] = datetime.now(timezone.utc)

    if not out.exists() or out.stat().st_size == 0:
        raise ValueError("No se pudo escribir el PDF.")
    return out
