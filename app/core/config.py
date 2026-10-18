from pathlib import Path

# Raíz de datos locales (secuencias, índices, modelos, visuales)
DATA_DIR = Path("data")
SEQUENCES_DIR = DATA_DIR / "Secuencias"
VISUALS_DIR = DATA_DIR / "Visuales"
MODELS_DIR = DATA_DIR / "models"
EXPORTS_DIR = DATA_DIR / "exports"

# Referencia miRBase (no va a la DB)
MIRBASE_MATURE_FILENAME = "mature.fa"

# Límites de entrada
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB por archivo
# Extensiones compuestas válidas (orden: más específicas primero)
ALLOWED_SUFFIXES = (
    ".fastq.gz",
    ".fasta.gz",
    ".fa.gz",
    ".fq.gz",
    ".fastq",
    ".fasta",
    ".fa",
    ".fq",
)
# Muestra rápida de estructura antes del QC completo (para GET temprano)
VALIDATION_PEEK_RECORDS = 200

# QC miRNA (maduros; 18–30 opcional para plantas/sequía)
DEFAULT_MIN_LEN_MIRNA = 18
DEFAULT_MAX_LEN_MIRNA = 25
EXTENDED_MAX_LEN_MIRNA = 30

# Chunk de lectura/escritura
IO_CHUNK_BYTES = 1024 * 1024  # 1 MB

# Etiquetado ML
LABEL_SOURCE_EVIDENCE = "evidence"
LABEL_SOURCE_SYNTHETIC_NEGATIVE = "synthetic_negative"
LABEL_SOURCE_UNLABELED = "unlabeled"


def user_dir(user_id: int) -> Path:
    path = SEQUENCES_DIR / f"user_{user_id}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_raw_dir(user_id: int) -> Path:
    path = user_dir(user_id) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_processed_dir(user_id: int) -> Path:
    path = user_dir(user_id) / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path
