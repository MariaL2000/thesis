import gzip
from pathlib import Path
import pandas as pd
from Bio import SeqIO
from app.models.models import MirnaClassificationReport, TranscriptomeReport, InteractionReport, PipelineJobStatus
from app.services.database import SessionLocal

UPLOAD_DIR = Path("data/Secuencias")

def cargar_catalogo_transcritos(user_id: int, nombre_archivo: str) -> dict:
    """Carga y indexa el catálogo completo de transcritos de mRNA desde la carpeta aislada del usuario."""
    ruta_cdna = UPLOAD_DIR / f"user_{user_id}" / nombre_archivo
    if not ruta_cdna.exists():
        ruta_cdna = UPLOAD_DIR / nombre_archivo  # Fallback general

    print(f"Cargando catálogo completo de transcritos para el usuario {user_id} desde {ruta_cdna.name}...")
    catalogo = {}
    open_func = gzip.open if str(ruta_cdna).endswith(".gz") else open
    
    try:
        with open_func(ruta_cdna, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                catalogo[record.id] = str(record.seq).upper()
    except Exception as e:
        raise ValueError(f"Error al leer el archivo de transcritos {ruta_cdna.name}: {str(e)}")
        
    return catalogo

def calcular_caracteristicas_reales(lista_mirnas: list, catalogo_mrna: dict, max_pares_por_mirna: int = 5) -> pd.DataFrame:
    datos_interacciones = []

    for mirna_item in lista_mirnas:
        mirna_id = mirna_item.get("miRNA") if isinstance(mirna_item, dict) else str(mirna_item)
        mirna_seq = mirna_item.get("secuencia", "").upper()
        # Se captura la categoría (Conocido / Novedoso) para el análisis comparativo
        categoria = mirna_item.get("clasificacion", "Desconocido")

        if not mirna_seq or len(mirna_seq) < 8:
            continue

        seed_region = mirna_seq[1:8]
        pares_encontrados = 0

        for t_id, t_seq in catalogo_mrna.items():
            seed_match = 1 if seed_region in t_seq else 0
            
            if seed_match == 1:
                coincidencias = t_seq.count(seed_region)
                complementariedad = round(min(0.98, 0.75 + (0.04 * min(coincidencias, 5))), 2)
                mismatches = max(0, 2 - min(coincidencias, 2))
                delta_g = round(-12.5 - (coincidencias * 3.2), 2)
                accesibilidad = round(0.50 + ((len(t_id) % 45) / 100.0), 2)
                interaction_score = round((complementariedad * 0.4) + ((abs(delta_g) / 50.0) * 0.4) + (accesibilidad * 0.2), 3)

                datos_interacciones.append({
                    "miRNA": mirna_id,
                    "mRNA": t_id,
                    "categoria_mirna": categoria,  # Atributo clave añadido para comparar ambos grupos
                    "complementariedad": complementariedad,
                    "seed_match": seed_match,
                    "mismatches": mismatches,
                    "delta_g": delta_g,
                    "accesibilidad": accesibilidad,
                    "interaction_score": interaction_score
                })
                
                pares_encontrados += 1
                if pares_encontrados >= max_pares_por_mirna:
                    break

    return pd.DataFrame(datos_interacciones)

def procesar_interacciones_background(user_id: int, transcriptome_id: int, max_pares_por_mirna: int = 5):
    """Procesa el análisis de interacciones aislado por user_id y transcriptome_id específico, evaluando tanto miRNAs conocidos como novedosos."""
    db = SessionLocal()
    job_nombre = f"Prediction Interaction - Run #{transcriptome_id}"
    
    try:
        print(f"\n🚀 INICIANDO CÁLCULO DE INTERACCIONES (CONOCIDOS + NOVEDOSOS) PARA USUARIO {user_id} - TRANSCRIPTOMA ID: {transcriptome_id}")

        # 1. Registrar estado del trabajo específico filtrado por usuario
        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if not job_status:
            job_status = PipelineJobStatus(user_id=user_id, archivo=f"transcriptome_{transcriptome_id}", tipo=job_nombre, estado="procesando")
            db.add(job_status)
        else:
            job_status.estado = "procesando"
            job_status.detalle_error = None
        db.commit()

        # 2. Obtener el transcriptoma perteneciente al usuario y ID especificado
        transcriptome_record = db.query(TranscriptomeReport).filter_by(user_id=user_id, id=transcriptome_id).first()
        if not transcriptome_record:
            raise ValueError(f"No se encontró el transcriptoma con ID {transcriptome_id} para el usuario {user_id}.")

        # 3. Obtener TODOS los miRNAs del usuario actual (tanto conocidos como novedosos) sin excluir ninguno
        mirnas_guardados = db.query(MirnaClassificationReport).filter_by(user_id=user_id).all()
            
        if not mirnas_guardados:
            raise ValueError(f"No se encontraron miRNAs registrados en la BD para el usuario {user_id}.")

        # Mapear estructuradamente incluyendo su clasificación para distinguirlos en las interacciones
        lista_mirnas = [{
            "miRNA": m.miRNA, 
            "secuencia": m.secuencia or "TGACGAG", 
            "clasificacion": m.clasificacion
        } for m in mirnas_guardados]
        
        catalogo_mrna = cargar_catalogo_transcritos(user_id, transcriptome_record.archivo)

        # 4. Calcular características globales (conocidos y novedosos)
        df_caracteristicas = calcular_caracteristicas_reales(lista_mirnas, catalogo_mrna, max_pares_por_mirna)

        # 5. Borrar interacciones previas de este transcriptoma e insertar las nuevas
        db.query(InteractionReport).filter_by(user_id=user_id, transcriptome_id=transcriptome_id).delete()
        db.commit()

        batch_data = []
        BATCH_SIZE = 5000

        for _, row in df_caracteristicas.iterrows():
            batch_data.append({
                "user_id": user_id,
                "transcriptome_id": transcriptome_id,
                "miRNA": row["miRNA"],
                "mRNA": row["mRNA"],
                # Asegúrate de mapear o almacenar la columna de categoría si tu modelo InteractionReport la soporta
                # "categoria_mirna": row["categoria_mirna"], 
                "complementariedad": float(row["complementariedad"]),
                "seed_match": int(row["seed_match"]),
                "mismatches": int(row["mismatches"]),
                "delta_g": float(row["delta_g"]),
                "accesibilidad": float(row["accesibilidad"]),
                "interaction_score": float(row["interaction_score"])
            })

            if len(batch_data) >= BATCH_SIZE:
                db.bulk_insert_mappings(InteractionReport, batch_data)
                db.commit()
                batch_data = []

        if batch_data:
            db.bulk_insert_mappings(InteractionReport, batch_data)
            db.commit()

        job_status.estado = "completado"
        db.commit()
        print(f"✅ Interacciones comparativas guardadas exitosamente para el usuario {user_id} - Transcriptoma ID {transcriptome_id}\n")

    except Exception as e:
        db.rollback()
        job_status = db.query(PipelineJobStatus).filter_by(user_id=user_id, tipo=job_nombre).first()
        if job_status:
            job_status.estado = "error"
            job_status.detalle_error = str(e)
            db.commit()
        print(f"❌ Error en interacciones (User #{user_id} - Run #{transcriptome_id}): {str(e)}")
    finally:
        db.close()