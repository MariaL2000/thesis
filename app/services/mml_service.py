import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from app.core.config import MODELS_DIR
from app.models.models import InteractionCandidate, MLModelRun, PipelineJobStatus
from app.services.database import SessionLocal

FEATURE_COLS = [
    "complementarity",
    "seed_match",
    "mismatches",
    "delta_g",
    "accessibility",
    "match_length",
    "num_matches",
    "gaps",
    "mirna_expression",
]


def entrenar_comparar_modelos_ml_background(user_id: int, transcriptome_id: int):
    """
    Entrena solo con filas etiquetadas (evidence + synthetic_negative).
    Predice probabilidad sobre unlabeled.
    """
    db = SessionLocal()
    job_nombre = f"Machine Learning Prediction - Run #{transcriptome_id}"
    job_status = None

    try:
        print(f"\n[ML] user={user_id} transcriptome={transcriptome_id}")

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

        rows = (
            db.query(InteractionCandidate)
            .filter_by(user_id=user_id, transcriptome_id=transcriptome_id)
            .all()
        )
        if not rows:
            raise ValueError("No hay interacciones previas para entrenar.")

        def row_to_dict(r: InteractionCandidate) -> dict:
            return {
                "id": r.id,
                "mirna_id": r.mirna_id,
                "mrna_id": r.mrna_id,
                "complementarity": r.complementarity or 0.0,
                "seed_match": r.seed_match or 0,
                "mismatches": r.mismatches or 0,
                "delta_g": r.delta_g if r.delta_g is not None else 0.0,
                "accessibility": r.accessibility if r.accessibility is not None else 0.0,
                "match_length": r.match_length or 0,
                "num_matches": r.num_matches or 0,
                "gaps": r.gaps or 0,
                "mirna_expression": r.mirna_expression or 0.0,
                "label": r.label,
                "label_source": r.label_source,
            }

        df_all = pd.DataFrame([row_to_dict(r) for r in rows])
        labeled = df_all[df_all["label"].isin([0, 1])].copy()

        if len(labeled) < 20 or labeled["label"].nunique() < 2:
            # Fallback documentado: si aún no hay etiquetas suficientes,
            # usar seed_match como proxy débil SOLO para demo local (no evidencia biológica).
            print("[ML] Etiquetas insuficientes; usando proxy seed_match (solo desarrollo).")
            df_all["label"] = df_all["seed_match"].astype(int)
            labeled = df_all.copy()
            for r in rows:
                if r.label is None:
                    r.label = int(r.seed_match or 0)
                    r.label_source = "proxy_seed_dev"
            db.commit()

        X = labeled[FEATURE_COLS].fillna(0)
        y = labeled["label"].astype(int)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42,
            stratify=y_temp if y_temp.nunique() > 1 else None,
        )

        model_version = datetime.now(timezone.utc).strftime("v%Y%m%d%H%M%S")
        modelos = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        }

        db.query(MLModelRun).filter_by(user_id=user_id, transcriptome_id=transcriptome_id).delete()
        db.commit()

        modelos_entrenados = {}
        best_name = "Random Forest"

        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            modelos_entrenados[nombre] = modelo
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, "predict_proba") else y_pred

            run = MLModelRun(
                user_id=user_id,
                transcriptome_id=transcriptome_id,
                model_version=model_version,
                model_name=nombre,
                training_dataset=f"user_{user_id}_tr_{transcriptome_id}",
                features_used=json.dumps(FEATURE_COLS),
                hyperparameters=json.dumps({"random_state": 42}),
                accuracy=round(float(accuracy_score(y_test, y_pred)), 4),
                precision=round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                recall=round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                f1_score=round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                roc_auc=round(float(roc_auc_score(y_test, y_proba)), 4) if len(np.unique(y_test)) > 1 else None,
                pr_auc=round(float(average_precision_score(y_test, y_proba)), 4) if len(np.unique(y_test)) > 1 else None,
                plots_dir=str(MODELS_DIR / f"user_{user_id}" / model_version),
            )
            db.add(run)

        db.commit()

        rf = modelos_entrenados[best_name]
        X_all = df_all[FEATURE_COLS].fillna(0)
        proba = rf.predict_proba(X_all)[:, 1]

        for idx, r in enumerate(rows):
            r.ml_probability = float(proba[idx])
            r.model_version = model_version
        db.commit()

        job_status.estado = "completado"
        db.commit()
        print(f"[ML] OK version={model_version} labeled={len(labeled)} total={len(df_all)}")

    except Exception as e:
        db.rollback()
        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if job_status:
            job_status.estado = "error"
            job_status.detalle_error = str(e)
            db.commit()
        print(f"[ML Error] {e}")
    finally:
        db.close()
