import gzip
import time
from pathlib import Path
from typing import Tuple, Set, Dict
from Bio import SeqIO
from sqlalchemy import func

from app.services.database import SessionLocal
from app.models.models import MirnaReport, MirnaClassificationReport, PipelineJobStatus

UPLOAD_DIR = Path("data/Secuencias")


def cargar_referencia_mirbase(referencia_filename: str = "mature.fa") -> Tuple[Dict[str, str], Set[str], Set[str]]:
    """Carga los IDs, secuencias exactas y regiones seed válidas desde el archivo local de miRBase."""
    ruta_ref = UPLOAD_DIR / referencia_filename
    if not ruta_ref.exists():
        ruta_ref = Path("data") / referencia_filename

    miRBase_ref_ids = {}
    miRBase_ref_seqs = set()
    miRBase_ref_seeds = set()
    
    if ruta_ref.exists():
        formato = "fastq" if "fastq" in referencia_filename.lower() or "fq" in referencia_filename.lower() else "fasta"
        open_func = gzip.open if referencia_filename.endswith(".gz") else open
        
        try:
            with open_func(ruta_ref, "rt") as f_ref:
                for record in SeqIO.parse(f_ref, formato):
                    if record.id:
                        clean_id = record.id.split()[0].upper()
                        miRBase_ref_ids[clean_id] = record.id.split()[0]
                    if record.seq:
                        seq_str = str(record.seq).upper()
                        miRBase_ref_seqs.add(seq_str)
                        if len(seq_str) >= 8:
                            miRBase_ref_seeds.add(seq_str[1:8]) # Región seed estándar (posiciones 2 a 8)
            print(f"-> Referencia '{referencia_filename}' cargada con éxito desde disco.")
        except Exception as e:
            raise ValueError(f"Error al leer el archivo de referencia {referencia_filename}: {str(e)}")
    else:
        # Fallback de seguridad si no existe el archivo físico
        miRBase_ref_ids = {"OSA-MIR156A": "osa-MIR156a", "OSA-MIR168A": "osa-MIR168a"}
        miRBase_ref_seqs = {"TGACGAG", "TCGCGG"}
        miRBase_ref_seeds = {"GACG", "CGCG"}
        
    return miRBase_ref_ids, miRBase_ref_seqs, miRBase_ref_seeds


def procesar_clasificacion_desde_db_background(user_id: int, referencia_filename: str = "mature.fa"):
    """
    Tarea en segundo plano que procesa los registros de un usuario específico comparándolos con mature.fa, 
    guarda los resultados (conocidos y novel) en la BD y actualiza el estado del job a 'completado'.
    """
    db = SessionLocal()
    try:
        print("\n" + "="*60)
        print(f"🚀 INICIANDO CLASIFICACIÓN CONTRA MIＲBASE PARA EL USUARIO ID: {user_id}")
        print("="*60)
        
        t_inicio = time.time()
        mirbase_ids_dict, mirbase_seqs, mirbase_seeds = cargar_referencia_mirbase(referencia_filename)

        # 1. Identificar la fecha de la última carga masiva en la BD para este usuario específico
        ultima_fecha = db.query(func.max(MirnaReport.fecha)).filter(MirnaReport.user_id == user_id).scalar()
        if not ultima_fecha:
            print(f"⚠️ [Background] No hay registros en la BD para clasificar del usuario {user_id}.")
            return

        # 2. Limpiar reportes de clasificación anteriores de este usuario para que no se mezclen
        db.query(MirnaClassificationReport).filter(MirnaClassificationReport.user_id == user_id).delete()
        db.commit()

        # 3. Filtrar SÓLO los registros que pertenecen a esa última tanda de este usuario
        registros_bd = db.query(MirnaReport).filter(
            MirnaReport.user_id == user_id,
            MirnaReport.fecha == ultima_fecha
        ).all()
        
        if not registros_bd:
            print(f"⚠️ [Background] No se encontraron registros recientes para la sesión actual del usuario {user_id}.")
            return

        print(f"📊 Total de registros recientes recuperados para clasificar (Usuario {user_id}): {len(registros_bd):,}")

        batch_data = []
        BATCH_SIZE = 5000
        total_procesados = 0

        for item in registros_bd:
            mirna_id = getattr(item, "miRNA", "Desconocido")
            secuencia = str(getattr(item, "secuencia", "")).upper().replace("N", "")
            nombre_archivo = getattr(item, "archivo", "desconocido")

            if not secuencia:
                continue

            estado = "Potencialmente Novedoso (Novel miRNA)"
            match_identificado = mirna_id

            # Algoritmo de comparación jerárquica con miRBase (mature.fa)
            if mirna_id.upper() in mirbase_ids_dict:
                real_id = mirbase_ids_dict[mirna_id.upper()]
                especie = real_id.split("-")[0].lower() if "-" in real_id else "global"
                estado = f"Conocido ({especie.upper()} - ID Exacto: {real_id})"
                match_identificado = real_id
            elif secuencia in mirbase_seqs:
                estado = "Conocido (miRBase - Coincidencia de Secuencia Exacta)"
            else:
                seed = secuencia[1:8] if len(secuencia) >= 8 else secuencia
                if seed in mirbase_seeds:
                    estado = "Conocido (miRBase - Homología por Región Seed)"

            # TODOS (conocidos y novel) se guardan estructurados en la BD
            batch_data.append({
                "user_id": user_id,
                "archivo": nombre_archivo,
                "miRNA": match_identificado,
                "secuencia": secuencia,
                "clasificacion": estado
            })

            total_procesados += 1
            if len(batch_data) >= BATCH_SIZE:
                db.bulk_insert_mappings(MirnaClassificationReport, batch_data)
                db.commit()
                batch_data = []

        if batch_data:
            db.bulk_insert_mappings(MirnaClassificationReport, batch_data)
            db.commit()

        # 4. ACTUALIZAR EL ESTADO DEL JOB A "completado" para este usuario
        db.query(PipelineJobStatus).filter(
            PipelineJobStatus.user_id == user_id,
            PipelineJobStatus.tipo == "miRBase Classification"
        ).update({"estado": "completado"})
        db.commit()

        t_total = time.time() - t_inicio
        print(f"\n✅ ¡Clasificación finalizada para el usuario {user_id}! {total_procesados:,} registros procesados en {t_total:.2f} segundos.\n")

    except Exception as e:
        db.rollback()
        db.query(PipelineJobStatus).filter(
            PipelineJobStatus.user_id == user_id,
            PipelineJobStatus.tipo == "miRBase Classification"
        ).update({"estado": "error", "detalle_error": str(e)})
        db.commit()
        print(f"❌ [Background] Error crítico en clasificación para el usuario {user_id}: {str(e)}")
    finally:
        db.close()



"""
los miRNAs conocidos (y los desconocidos/novel) se guardan exactamente
igual en la base de datos dentro de la tabla MirnaClassificationReport. 
La única diferencia es que el campo clasificacion almacena una etiqueta 
descriptiva (por ejemplo, Conocido (HSA - ID Exacto: hsa-miR-21-5p) o
 Potencialmente Novedoso (Novel miRNA)) según el resultado de la comparación contra mature.fa.
 """
