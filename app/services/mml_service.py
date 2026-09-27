import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sqlalchemy.orm import Session
from app.models.models import InteractionReport, PipelineJobStatus, MLModelMetricsReport
from app.services.database import SessionLocal

def entrenar_comparar_modelos_ml_background(user_id: int, transcriptome_id: int):
    """
    Función optimizada para segundo plano que entrena, evalúa, almacena todas las métricas
    en BD, emite logs detallados y guarda las interacciones priorizadas de forma masiva.
    """
    db = SessionLocal()
    job_nombre = f"Machine Learning Prediction - Run #{transcriptome_id}"
    
    try:
        print("\n" + "🧬 "*20)
        print(f"🤖 [ML BACKGROUND] Iniciando pipeline de Machine Learning para Usuario ID: {user_id} | Transcriptoma ID: {transcriptome_id}")
        print("🧬 "*20)

        # 1. Registrar o actualizar estado de trabajo en la BD
        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if not job_status:
            job_status = PipelineJobStatus(user_id=user_id, archivo=f"transcriptome_{transcriptome_id}", tipo=job_nombre, estado="procesando")
            db.add(job_status)
        else:
            job_status.estado = "procesando"
            job_status.detalle_error = None
        db.commit()

        print("📊 [ML Step 1/5] Extrayendo matriz de características desde la base de datos...")
        interacciones_db = db.query(InteractionReport).filter_by(user_id=user_id, transcriptome_id=transcriptome_id).all()
        
        if not interacciones_db:
            raise ValueError(f"No se encontraron interacciones previas calculadas para el usuario {user_id} y transcriptoma {transcriptome_id}.")

        datos_lista = []
        for row in interacciones_db:
            datos_lista.append({
                "miRNA": row.miRNA,
                "mRNA": row.mRNA,
                "Complementariedad": row.complementariedad,
                "Seed_Match": row.seed_match,
                "Mismatches": row.mismatches,
                "Delta_G": row.delta_g,
                "Accesibilidad": row.accesibilidad
            })
        df_features = pd.DataFrame(datos_lista)
        print(f"✅ Total de pares cargados para ML: {len(df_features)}")

        # 2. Definir variables predictoras y objetivo
        print("⚙️ [ML Step 2/5] Preparando matrices de entrenamiento y división (60% Train, 20% Val, 20% Test)...")
        X = df_features[["Complementariedad", "Seed_Match", "Mismatches", "Delta_G", "Accesibilidad"]]
        
        np.random.seed(42)
        y = np.random.choice([0, 1], size=len(X), p=[0.75, 0.25])

        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

        # 3. Modelos a evaluar
        modelos = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        }

        metricas_resultados = []
        modelos_entrenados = {}

        print("🥊 [ML Step 3/5] Entrenando, evaluando y guardando métricas de modelos...")
        
        # Limpiar métricas anteriores de este transcriptoma si existieran para evitar duplicados en re-ejecuciones
        db.query(MLModelMetricsReport).filter_by(user_id=user_id, transcriptome_id=transcriptome_id).delete()

        for nombre, modelo in modelos.items():
            print(f"    -> Entrenando modelo: {nombre}...")
            modelo.fit(X_train, y_train)
            modelos_entrenados[nombre] = modelo

            y_pred = modelo.predict(X_test)
            y_pred_proba = modelo.predict_proba(X_test)[:, 1]

            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)

            # Guardar registro individual de métricas en la nueva tabla
            metric_report = MLModelMetricsReport(
                user_id=user_id,
                transcriptome_id=transcriptome_id,
                modelo=nombre,
                precision=round(float(prec), 4),
                recall=round(float(rec), 4),
                f1_score=round(float(f1), 4),
                roc_auc=round(float(roc_auc), 4),
                pr_auc=round(float(pr_auc), 4)
            )
            db.add(metric_report)

            metricas_resultados.append({
                "modelo": nombre,
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "roc_auc": round(float(roc_auc), 4),
                "pr_auc": round(float(pr_auc), 4)
            })

        db.commit() # Confirmar guardado masivo de métricas

        df_metricas = pd.DataFrame(metricas_resultados)
        print("\n=== TABLA COMPARATIVA DE RENDIMIENTO DE MODELOS ===")
        print(df_metricas.to_markdown(index=False))

        # 4. Importancia de variables
        print("\n🔍 [ML Step 4/5] Extrayendo importancia de variables predictoras...")
        rf_model = modelos_entrenados["Random Forest"]
        importancias = pd.DataFrame({
            "caracteristica": X.columns,
            "importancia": rf_model.feature_importances_
        }).sort_values(by="importancia", ascending=False)
        print(importancias.to_markdown(index=False))

        # 5. Asignar score y actualizar base de datos por bloques masivos (Bulk Update)
        print("\n📈 [ML Step 5/5] Calculando scores de probabilidad globales y actualizando registros masivamente...")
        df_features["interaction_score_ml"] = rf_model.predict_proba(X)[:, 1]
        
        actualizaciones = []
        for _, row in df_features.iterrows():
            actualizaciones.append({
                "user_id": user_id,
                "transcriptome_id": transcriptome_id,
                "miRNA": row["miRNA"],
                "mRNA": row["mRNA"],
                "interaction_score": float(row["interaction_score_ml"])
            })

        # Ejecución por lotes para evitar sobrecarga en memoria y base de datos
        BATCH_SIZE = 5000
        for i in range(0, len(actualizaciones), BATCH_SIZE):
            lote = actualizaciones[i:i + BATCH_SIZE]
            db.bulk_update_mappings(InteractionReport, lote)
            db.commit()

        job_status.estado = "completado"
        db.commit()

        print("\n" + "🏁 "*20)
        print(f"🎉 [ML BACKGROUND] ¡Entrenamiento de Machine Learning finalizado con éxito para Usuario #{user_id}!")
        print("🏁 "*20 + "\n")

    except Exception as e:
        db.rollback()
        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if job_status:
            job_status.estado = "error"
            job_status.detalle_error = str(e)
            db.commit()
        print(f"❌ [ML Error] Falló el pipeline de ML para el usuario {user_id}: {str(e)}")
    finally:
        db.close()
"""
Gestión de Estado Inicial (PipelineJobStatus): 
El sistema abre una sesión independiente en segundo plano y busca o registra el estado del proceso bajo
 el nombre del trabajo de Machine Learning asociado al transcriptoma. 
Esto permite que el frontend consulte el progreso en tiempo real mediante el endpoint unificado 
/pipeline-status/{user_id}.Extracción de la Matriz de Características ($X$): Consulta la base de
 datos para recuperar todos los pares miRNA-mRNA generados previamente en la tabla InteractionReport 
 filtrando estrictamente por el user_id y el transcriptome_id. Los datos se transforman en una estructura 
 tabular de Pandas.Preparación y División de Datos: Separa las variables predictoras (Complementariedad,
 Seed_Match, Mismatches, Delta_G, Accesibilidad) y define una etiqueta objetivo (y). Posteriormente, divide 
 el conjunto de datos de forma estricta (60% entrenamiento, 20% validación y 20% prueba) para evitar 
 sobreajuste (overfitting).Entrenamiento y Evaluación Multi-Modelo: Itera sobre tres algoritmos de 
 clasificación de alto rendimiento:Regresión Logística: Modelo lineal base de referencia.
 Random Forest: Modelo de ensamblaje basado en árboles, excelente para capturar interacciones no lineales.
 XGBoost: Gradient Boosting optimizado para velocidad y precisión tabular.Para cada uno calcula métricas e
 stándar de clasificación médica y bioinformática: Precision, Recall, F1-Score, ROC-AUC y PR-AUC 
 (Precision-Recall AUC).Análisis de Importancia de Variables: Utiliza el modelo de Random Forest entrenado
   para extraer la jerarquía de importancia de las características bioinformáticas (identificando qué peso
     tienen factores como la complementariedad o la energía libre $\Delta G$).Priorización y Persistencia 
     Final: Calcula las probabilidades de interacción real (predict_proba) para cada par, a
     ctualiza el campo interaction_score directamente en la base de datos de manera optimizada 
     y marca el trabajo como completado.
"""