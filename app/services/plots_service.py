"""Gráficos Matplotlib del análisis de la búsqueda actual."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from app.core.config import VISUALS_DIR


def generar_graficos_analisis(
    user_id: int,
    transcriptome_id: int,
    modelos_entrenados: dict,
    X_test: pd.DataFrame,
    y_test,
    df_busqueda: pd.DataFrame,
) -> str:
    ruta_base = os.path.join(VISUALS_DIR, f"user_{user_id}_run_{transcriptome_id}")
    os.makedirs(ruta_base, exist_ok=True)
    y_test = np.asarray(y_test)

    if len(np.unique(y_test)) > 1:
        plt.figure(figsize=(8, 6))
        for nombre, modelo in modelos_entrenados.items():
            if not hasattr(modelo, "predict_proba"):
                continue
            proba = modelo.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc_val = roc_auc_score(y_test, proba)
            plt.plot(fpr, tpr, label=f"{nombre} (AUC = {auc_val:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Azar (AUC = 0.500)")
        plt.xlabel("Tasa de falsos positivos")
        plt.ylabel("Tasa de verdaderos positivos")
        plt.title(f"Curvas ROC — usuario {user_id}")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_base, "curva_roc_comparativa.png"), dpi=160)
        plt.close()

        plt.figure(figsize=(8, 6))
        for nombre, modelo in modelos_entrenados.items():
            if not hasattr(modelo, "predict_proba"):
                continue
            proba = modelo.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, proba)
            pr_auc = average_precision_score(y_test, proba)
            plt.plot(recall, precision, label=f"{nombre} (AP = {pr_auc:.3f})", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall — transcriptoma {transcriptome_id}")
        plt.legend(loc="lower left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_base, "curva_precision_recall.png"), dpi=160)
        plt.close()

    rf_model = modelos_entrenados.get("Random Forest")
    if rf_model is not None and hasattr(rf_model, "feature_importances_"):
        plt.figure(figsize=(9, 5))
        importancias = np.asarray(rf_model.feature_importances_)
        caracteristicas = list(X_test.columns)
        indices = np.argsort(importancias)
        plt.barh(range(len(indices)), importancias[indices], color="steelblue", edgecolor="navy")
        plt.yticks(range(len(indices)), [caracteristicas[i] for i in indices])
        plt.xlabel("Importancia (Gini)")
        plt.title("Qué features usa Random Forest")
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_base, "importancia_variables.png"), dpi=160)
        plt.close()

    if df_busqueda is not None and len(df_busqueda) and "complementarity" in df_busqueda.columns:
        comp = pd.to_numeric(df_busqueda["complementarity"], errors="coerce").dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(comp, bins=20, color="#2a6f97", edgecolor="white")
        plt.xlabel("Complementariedad (Watson-Crick + 0.5·wobble)")
        plt.ylabel("Pares miRNA–mRNA")
        plt.title("Distribución de complementariedad")
        plt.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_base, "complementariedad_hist.png"), dpi=160)
        plt.close()

    if df_busqueda is not None and {"delta_g", "complementarity"}.issubset(df_busqueda.columns):
        plt.figure(figsize=(8, 6))
        x = pd.to_numeric(df_busqueda["delta_g"], errors="coerce")
        y = pd.to_numeric(df_busqueda["complementarity"], errors="coerce")
        c = pd.to_numeric(df_busqueda.get("ml_probability", 0.5), errors="coerce")
        sc = plt.scatter(x, y, c=c, cmap="viridis", alpha=0.85, edgecolors="none")
        plt.colorbar(sc, label="Probabilidad del modelo")
        plt.xlabel("ΔG estimado (kcal/mol)")
        plt.ylabel("Complementariedad")
        plt.title("Energía de unión frente a complementariedad")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_base, "delta_g_vs_complementariedad.png"), dpi=160)
        plt.close()

    if df_busqueda is not None and "seed_match" in df_busqueda.columns:
        conteo = df_busqueda["seed_match"].fillna(0).astype(int).value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        plt.bar([str(i) for i in conteo.index], conteo.values, color=["#c45c26", "#1b7f4e"][: len(conteo)])
        plt.xlabel("Seed match (0 = no, 1 = sí)")
        plt.ylabel("Pares")
        plt.title("Sitios con semilla complementaria")
        plt.tight_layout()
        plt.savefig(os.path.join(ruta_base, "seed_match_conteo.png"), dpi=160)
        plt.close()

    return ruta_base
