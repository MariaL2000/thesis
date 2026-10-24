import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from app.core.config import VISUALS_DIR
from app.models.models import InteractionCandidate
from app.services.database import SessionLocal
from app.services.mml_service import entrenar_comparar_modelos_ml_background as _train_core

VISUAL_DIR = VISUALS_DIR


def generar_y_guardar_graficos_ml(modelos_entrenados, X_test, y_test, user_id, transcriptome_id):
    os.makedirs(VISUAL_DIR, exist_ok=True)
    ruta_base = os.path.join(VISUAL_DIR, f"user_{user_id}_run_{transcriptome_id}")
    os.makedirs(ruta_base, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for nombre, modelo in modelos_entrenados.items():
        if hasattr(modelo, "predict_proba") and len(np.unique(y_test)) > 1:
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_val = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{nombre} (AUC = {auc_val:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.500)")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title(f"Curvas ROC - Usuario #{user_id}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(ruta_base, "curva_roc_comparativa.png"), dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    for nombre, modelo in modelos_entrenados.items():
        if hasattr(modelo, "predict_proba") and len(np.unique(y_test)) > 1:
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            plt.plot(recall, precision, label=f"{nombre} (PR-AUC = {pr_auc:.3f})", linewidth=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall - Transcriptoma #{transcriptome_id}")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.savefig(os.path.join(ruta_base, "curva_precision_recall.png"), dpi=300, bbox_inches="tight")
    plt.close()

    rf_model = modelos_entrenados.get("Random Forest")
    if rf_model and hasattr(rf_model, "feature_importances_"):
        plt.figure(figsize=(9, 5))
        importancias = rf_model.feature_importances_
        caracteristicas = list(X_test.columns)
        indices = np.argsort(importancias)
        plt.barh(range(len(indices)), importancias[indices], color="steelblue", edgecolor="navy")
        plt.yticks(range(len(indices)), [caracteristicas[i] for i in indices])
        plt.xlabel("Importancia (Gini)")
        plt.title("Importancia de características (Random Forest)")
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.savefig(os.path.join(ruta_base, "importancia_variables.png"), dpi=300, bbox_inches="tight")
        plt.close()

    return ruta_base


def validar_predicciones_biologicamente(df_res: pd.DataFrame, ruta_dataset_externo: str = None) -> pd.DataFrame:
    datos_externos = set()
    if ruta_dataset_externo and os.path.exists(ruta_dataset_externo):
        df_ext = pd.read_csv(ruta_dataset_externo)
        col = "miRNA" if "miRNA" in df_ext.columns else df_ext.columns[0]
        datos_externos = set(df_ext[col].values)

    evidencias = []
    for _, row in df_res.iterrows():
        score = row.get("ml_probability", 0.5)
        mirna = row["mirna_id"]
        delta_g = row.get("delta_g", 0)
        seed_match = row.get("seed_match", 0)
        if mirna in datos_externos:
            evidencias.append("Alta (dataset externo)")
        elif score > 0.85 and seed_match == 1 and (delta_g is not None and delta_g < -22.0):
            evidencias.append("Alta (modelo + termodinámica)")
        elif score > 0.65:
            evidencias.append("Media")
        else:
            evidencias.append("Baja")
    df_res["evidence"] = evidencias
    return df_res


def entrenar_comparar_modelos_ml_background(
    user_id: int, transcriptome_id: int, ruta_dataset_externo: str = None
):
    """Pipeline ML + gráficos. Reutiliza entrenamiento core y añade visuales/evidencia."""
    # 1) Entrenar y persistir métricas/probabilidades
    _train_core(user_id, transcriptome_id)

    db = SessionLocal()
    try:
        rows = (
            db.query(InteractionCandidate)
            .filter_by(user_id=user_id, transcriptome_id=transcriptome_id)
            .all()
        )
        if not rows:
            return

        df = pd.DataFrame(
            [
                {
                    "id": r.id,
                    "mirna_id": r.mirna_id,
                    "delta_g": r.delta_g if r.delta_g is not None else 0.0,
                    "seed_match": r.seed_match or 0,
                    "ml_probability": r.ml_probability or 0.0,
                }
                for r in rows
            ]
        )
        df = validar_predicciones_biologicamente(df, ruta_dataset_externo)
        for _, row in df.iterrows():
            db.query(InteractionCandidate).filter_by(id=int(row["id"])).update(
                {"evidence": str(row.get("evidence", ""))}
            )
        db.commit()
    except Exception as e:
        print(f"[validation pipeline] {e}")
    finally:
        db.close()
