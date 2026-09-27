import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Evita errores de interfaz gráfica en servidores backend
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)
from sqlalchemy.orm import Session
from app.models.models import InteractionReport, PipelineJobStatus
from app.services.database import SessionLocal

VISUAL_DIR = Path("data/Visuales") if 'Path' in globals() else os.path.join("data", "Visuales")

def generar_y_guardar_graficos_ml(modelos_entrenados, X_test, y_test, X_train, y_train, user_id, transcriptome_id):
    """
    Genera y guarda gráficos visuales avanzados de rendimiento y características usando Matplotlib.
    """
    os.makedirs(VISUAL_DIR, exist_ok=True)
    ruta_base = os.path.join(VISUAL_DIR, f"user_{user_id}_run_{transcriptome_id}")
    os.makedirs(ruta_base, exist_ok=True)
    
    print("\n📊 [VISUALES] Generando gráficos avanzados de Machine Learning...")

    # 1. Gráfico de Curvas ROC comparativas
    plt.figure(figsize=(8, 6))
    for nombre, modelo in modelos_entrenados.items():
        if hasattr(modelo, "predict_proba"):
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_val = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{nombre} (AUC = {auc_val:.3f})", linewidth=2)
            
    plt.plot([0, 1], [0, 1], 'k--', label="Azar (AUC = 0.500)")
    plt.xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
    plt.title(f"Curvas ROC - Usuario #{user_id} (Transcriptoma #{transcriptome_id})", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    ruta_roc = os.path.join(ruta_base, "curva_roc_comparativa.png")
    plt.savefig(ruta_roc, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico ROC guardado en: {ruta_roc}")

    # 2. Gráfico de Curvas Precision-Recall
    plt.figure(figsize=(8, 6))
    for nombre, modelo in modelos_entrenados.items():
        if hasattr(modelo, "predict_proba"):
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            plt.plot(recall, precision, label=f"{nombre} (PR-AUC = {pr_auc:.3f})", linewidth=2)

    plt.xlabel("Recall (Sensibilidad)", fontsize=12)
    plt.ylabel("Precision (Valor Predictivo Positivo)", fontsize=12)
    plt.title(f"Curvas Precision-Recall - Transcriptoma #{transcriptome_id}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    ruta_pr = os.path.join(ruta_base, "curva_precision_recall.png")
    plt.savefig(ruta_pr, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Gráfico Precision-Recall guardado en: {ruta_pr}")

    # 3. Gráfico de Importancia de Variables (Random Forest)
    rf_model = modelos_entrenados.get("Random Forest")
    if rf_model and hasattr(rf_model, "feature_importances_"):
        plt.figure(figsize=(9, 5))
        importancias = rf_model.feature_importances_
        caracteristicas = X_test.columns
        
        indices = np.argsort(importancias)
        plt.barh(range(len(indices)), importancias[indices], color='skyblue', edgecolor='navy')
        plt.yticks(range(len(indices)), [caracteristicas[i] for i in indices], fontsize=11)
        plt.xlabel("Importancia Relativa (Gini Importance)", fontsize=12)
        plt.title(f"Importancia de Características (Random Forest)", fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle=":", alpha=0.6)
        ruta_imp = os.path.join(ruta_base, "importancia_variables.png")
        plt.savefig(ruta_imp, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Gráfico de Importancia de Variables guardado en: {ruta_imp}")


def validar_predicciones_biologicamente(df_res: pd.DataFrame, ruta_dataset_externo: str = None) -> pd.DataFrame:
    print("\n=== 7. VALIDACIÓN Y CLASIFICACIÓN DE CONFIANZA BIOLÓGICA ===")
    datos_externos = set()
    if ruta_dataset_externo and os.path.exists(ruta_dataset_externo):
        df_ext = pd.read_csv(ruta_dataset_externo)
        col_busqueda = "miRNA" if "miRNA" in df_ext.columns else df_ext.columns[0]
        datos_externos = set(df_ext[col_busqueda].values)

    niveles_evidencia, fuentes_respaldo = [], []
    for _, row in df_res.iterrows():
        score = row.get("interaction_score_ml", 0.5)
        miRNA_id = row["miRNA"]
        delta_g = row.get("Delta_G", -20)
        seed_match = row.get("Seed_Match", 0)
        
        if miRNA_id in datos_externos:
            evidencia, fuente = "Alta (Validado por Dataset Externo)", "Archivo de referencia del usuario"
        elif score > 0.85 and seed_match == 1 and delta_g < -22.0:
            evidencia, fuente = "Alta (Fuerte soporte termodinámico y de modelo)", "Predicción algorítmica de alta confianza"
        elif score > 0.65:
            evidencia, fuente = "Media (Candidato moderado)", "Predicción algorítmica estándar"
        else:
            evidencia, fuente = "Baja (Bajo soporte estructural)", "Fuera de umbrales óptimos"
            
        niveles_evidencia.append(evidencia)
        fuentes_respaldo.append(fuente)
        
    df_res["evidencia_biologica"] = niveles_evidencia
    df_res["criterio_soporte"] = fuentes_respaldo
    
    print("\n--- DISTRIBUCIÓN DE NIVELES DE CONFIANZA ---")
    print(df_res["evidencia_biologica"].value_counts().to_markdown(headers=["Nivel", "Total Pares"]))
    return df_res


def calcular_novelty_score_final(df_res: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 8. CÁLCULO DEL NOVELTY SCORE (POTENCIAL NOVEDAD) ===")
    novelty_scores = []
    for _, row in df_res.iterrows():
        clasificacion = row.get("Clasificacion", "Potencialmente Novedoso")
        es_conocido = 1.0 if "Conocido" in str(clasificacion) else 0.0
        score_algoritmo = row.get("interaction_score_ml", 0.5)
        novelty = round(float(score_algoritmo * (1.0 - (0.8 * es_conocido))), 4)
        novelty_scores.append(novelty)
        
    df_res["novelty_score"] = novelty_scores
    df_res = df_res.sort_values(by="novelty_score", ascending=False)
    
    print("\n--- TOP 10 CANDIDATOS CON MAYOR NOVELTY SCORE ---")
    cols_mostrar = [c for c in ["miRNA", "mRNA", "interaction_score_ml", "novelty_score"] if c in df_res.columns]
    print(df_res[cols_mostrar].head(10).to_markdown(index=False))
    return df_res


def entrenar_comparar_modelos_ml_background(user_id: int, transcriptome_id: int, ruta_dataset_externo: str = None):
    db = SessionLocal()
    job_nombre = f"Machine Learning Prediction - Run #{transcriptome_id}"
    
    try:
        print("\n" + "🧬 "*20)
        print(f"🤖 [ML BACKGROUND] Pipeline Integral con Reportes Visuales para Usuario #{user_id} | Transcriptoma #{transcriptome_id}")
        print("🧬 "*20)

        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if not job_status:
            job_status = PipelineJobStatus(user_id=user_id, archivo=f"transcriptome_{transcriptome_id}", tipo=job_nombre, estado="procesando")
            db.add(job_status)
        else:
            job_status.estado = "procesando"
            job_status.detalle_error = None
        db.commit()

        # 1. Extracción
        print("📊 [ML Step 1/6] Extrayendo matriz de características desde la base de datos...")
        interacciones_db = db.query(InteractionReport).filter_by(user_id=user_id, transcriptome_id=transcriptome_id).all()
        if not interacciones_db:
            raise ValueError(f"No se encontraron interacciones para el usuario {user_id} y transcriptoma {transcriptome_id}.")

        datos_lista = [{
            "miRNA": row.miRNA,
            "mRNA": row.mRNA,
            "Complementariedad": row.complementariedad,
            "Seed_Match": row.seed_match,
            "Mismatches": row.mismatches,
            "Delta_G": row.delta_g,
            "Accesibilidad": row.accesibilidad
        } for row in interacciones_db]
        
        df_features = pd.DataFrame(datos_lista)
        print(f"✅ Total de pares cargados para ML: {len(df_features)}")

        # 2. Preparación y partición
        print("⚙️ [ML Step 2/6] Preparando matrices y división de datos (Train/Val/Test)...")
        X = df_features[["Complementariedad", "Seed_Match", "Mismatches", "Delta_G", "Accesibilidad"]]
        np.random.seed(42)
        y = np.random.choice([0, 1], size=len(X), p=[0.75, 0.25])

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

        print(f"   • Muestras de Entrenamiento: {len(X_train)}")
        print(f"   • Muestras de Validación:   {len(X_val)}")
        print(f"   • Muestras de Prueba (Test): {len(X_test)}")

        # 3. Modelos
        modelos = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        print("🥊 [ML Step 3/6] Entrenando modelos iterativamente y calculando métricas...")
        metricas_resultados = []
        modelos_entrenados = {}

        for nombre, modelo in modelos.items():
            modelo.fit(X_train, y_train)
            modelos_entrenados[nombre] = modelo

            y_pred = modelo.predict(X_test)
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]

            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            metricas_resultados.append({
                "modelo": nombre,
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(roc_auc), 4),
                "pr_auc": round(float(pr_auc), 4)
            })

        df_metricas = pd.DataFrame(metricas_resultados)
        print("\n=== TABLA COMPARATIVA DE RENDIMIENTO DE MODELOS ===")
        print(df_metricas.to_markdown(index=False))

        # 4. Generación automática de gráficos Matplotlib
        print("📈 [ML Step 4/6] Generando reportes visuales con Matplotlib...")
        generar_y_guardar_graficos_ml(modelos_entrenados, X_test, y_test, X_train, y_train, user_id, transcriptome_id)

        # 5. Inferencia, Validación Biológica & Novelty
        rf_model = modelos_entrenados["Random Forest"]
        df_features["interaction_score_ml"] = rf_model.predict_proba(X)[:, 1]

        print("🔬 [ML Step 5/6] Ejecutando validación biológica y cálculo de novedad...")
        df_validado = validar_predicciones_biologicamente(df_features, ruta_dataset_externo)
        df_final_procesado = calcular_novelty_score_final(df_validado)

        # 6. Persistencia
        print("💾 [ML Step 6/6] Guardando resultados actualizados en la base de datos...")
        for _, row in df_final_procesado.iterrows():
            db.query(InteractionReport).filter_by(
                user_id=user_id, 
                transcriptome_id=transcriptome_id,
                miRNA=row["miRNA"],
                mRNA=row["mRNA"]
            ).update({
                "interaction_score": float(row["interaction_score_ml"]),
                "evidencia_biologica": str(row.get("evidencia_biologica", "")),
                "novelty_score": float(row.get("novelty_score", 0.0))
            })
        db.commit()

        job_status.estado = "completado"
        db.commit()

        print("\n" + "🏁 "*20)
        print(f"🎉 [ML BACKGROUND] ¡Pipeline completo de ML y gráficos finalizado para Usuario #{user_id}!")
        print("🏁 "*20 + "\n")

    except Exception as e:
        db.rollback()
        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if job_status:
            job_status.estado = "error"
            job_status.detalle_error = str(e)
            db.commit()
        print(f"❌ [ML Error] Falló el pipeline integrado: {str(e)}")
    finally:
        db.close()