"""
Esquema DB mínimo: metadatos, resúmenes, matches e interacciones.
Las secuencias crudas/colapsadas y el transcriptoma viven en disco.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.services.database import Base


def _utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    input_files = relationship("InputFile", back_populates="user", cascade="all, delete-orphan")
    validation_reports = relationship("ValidationReport", back_populates="user", cascade="all, delete-orphan")
    mirna_summaries = relationship("MirnaInputSummary", back_populates="user", cascade="all, delete-orphan")
    identified_mirnas = relationship("IdentifiedMirna", back_populates="user", cascade="all, delete-orphan")
    transcriptomes = relationship("TranscriptomeReport", back_populates="user", cascade="all, delete-orphan")
    transcripts = relationship("TranscriptRecord", back_populates="user", cascade="all, delete-orphan")
    interactions = relationship("InteractionCandidate", back_populates="user", cascade="all, delete-orphan")
    job_statuses = relationship("PipelineJobStatus", back_populates="user", cascade="all, delete-orphan")
    ml_runs = relationship("MLModelRun", back_populates="user", cascade="all, delete-orphan")


class InputFile(Base):
    """Metadatos del archivo subido (ruta en disco, no contenido)."""

    __tablename__ = "input_files"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    role = Column(String(32), nullable=False)  # mirna | transcriptome
    original_filename = Column(String(255), nullable=False)
    stored_path = Column(String(512), nullable=False)
    format = Column(String(32), nullable=True)  # fasta | fastq | fastq.gz | ...
    size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="input_files")

    __table_args__ = (
        Index("idx_input_user_role", "user_id", "role"),
    )


class ValidationReport(Base):
    """Resultado de validación inicial (formato, tamaño, estructura). Consultable por GET."""

    __tablename__ = "validation_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    input_file_id = Column(Integer, ForeignKey("input_files.id", ondelete="CASCADE"), index=True, nullable=True)
    role = Column(String(32), nullable=False)
    filename = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False, default="procesando")  # procesando | completado | error
    format_ok = Column(Boolean, default=False)
    size_ok = Column(Boolean, default=False)
    structure_ok = Column(Boolean, default=False)
    looks_like_mirna = Column(Boolean, nullable=True)  # solo entrada 1
    total_records_sampled = Column(Integer, default=0)
    invalid_records = Column(Integer, default=0)
    avg_quality = Column(Float, nullable=True)  # FASTQ
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="validation_reports")


class MirnaInputSummary(Base):
    """
    Resumen del QC/colapsado de Entrada 1.
    Las lecturas colapsadas (seq → count) viven en collapsed_path (disco).
    """

    __tablename__ = "mirna_input_summaries"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    archivo = Column(String(255), index=True, nullable=False)
    collapsed_path = Column(String(512), nullable=False)
    total_reads = Column(Integer, default=0)
    reads_after_qc = Column(Integer, default=0)
    unique_sequences = Column(Integer, default=0)
    discarded = Column(Integer, default=0)
    min_len = Column(Integer, nullable=True)
    max_len = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="mirna_summaries")


class IdentifiedMirna(Base):
    """
    miRNA identificados contra mature.fa (y novel opcionales top-N).
    No guarda todas las lecturas colapsadas.
    """

    __tablename__ = "identified_mirnas"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    mirna_id = Column(String(100), index=True, nullable=False)  # nombre local / read id representativo
    mirbase_id = Column(String(100), index=True, nullable=True)
    mirna_sequence = Column(String(80), nullable=False)
    species = Column(String(32), nullable=True)
    read_count = Column(Integer, default=1)
    match_type = Column(String(64), nullable=False)  # exact | seed | mismatch | novel
    mismatches = Column(Integer, default=0)
    alignment_score = Column(Float, default=1.0)
    source_file = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="identified_mirnas")

    __table_args__ = (
        Index("idx_ident_user_mirbase", "user_id", "mirbase_id"),
        Index("idx_ident_user_seq", "user_id", "mirna_sequence"),
    )


class TranscriptomeReport(Base):
    """Resumen de Entrada 2. Secuencias en disco; IDs en TranscriptRecord."""

    __tablename__ = "transcriptome_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    fecha = Column(DateTime, default=_utcnow)
    archivo = Column(String(255), index=True, nullable=False)
    sequences_path = Column(String(512), nullable=True)  # copia/enlace lógico al FASTA en disco
    index_path = Column(String(512), nullable=True)  # TSV id\tgene_id\tlength\toffset
    total_transcritos = Column(Integer, default=0)
    transcritos_validos = Column(Integer, default=0)
    registros_invalidos_descartados = Column(Integer, default=0)
    longitud_minima_bp = Column(Integer, nullable=True)
    longitud_maxima_bp = Column(Integer, nullable=True)
    longitud_promedio_bp = Column(Float, nullable=True)
    longitud_mediana_bp = Column(Float, nullable=True)
    top_prefijo = Column(String(64), nullable=True)
    top_prefijo_conteo = Column(Integer, nullable=True)

    user = relationship("User", back_populates="transcriptomes")
    transcripts = relationship("TranscriptRecord", back_populates="transcriptome", cascade="all, delete-orphan")
    interactions = relationship("InteractionCandidate", back_populates="transcriptome", cascade="all, delete-orphan")
    ml_runs = relationship("MLModelRun", back_populates="transcriptome", cascade="all, delete-orphan")


class TranscriptRecord(Base):
    """Metadatos de transcrito (sin secuencia). Secuencia se lee por offset desde disco."""

    __tablename__ = "transcript_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    transcriptome_id = Column(
        Integer, ForeignKey("transcriptome_reports.id", ondelete="CASCADE"), index=True, nullable=False
    )
    transcript_id = Column(String(120), index=True, nullable=False)
    gene_id = Column(String(120), index=True, nullable=True)
    length_bp = Column(Integer, nullable=False)
    file_offset = Column(Integer, nullable=True)  # offset en sequences_path si aplica
    annotation = Column(String(512), nullable=True)

    user = relationship("User", back_populates="transcripts")
    transcriptome = relationship("TranscriptomeReport", back_populates="transcripts")

    __table_args__ = (
        UniqueConstraint("transcriptome_id", "transcript_id", name="uq_transcript_per_report"),
        Index("idx_tr_user_gene", "user_id", "gene_id"),
    )


class InteractionCandidate(Base):
    """
    Par miRNA–mRNA + features (X) + etiqueta (Y) + predicción.
    Dataset de entrenamiento / inferencia. No guarda secuencias largas.
    """

    __tablename__ = "interaction_candidates"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    transcriptome_id = Column(
        Integer, ForeignKey("transcriptome_reports.id", ondelete="CASCADE"), index=True, nullable=False
    )

    mirna_id = Column(String(100), index=True, nullable=False)
    mrna_id = Column(String(120), index=True, nullable=False)
    gene_id = Column(String(120), index=True, nullable=True)

    # Features
    seed_match = Column(Integer, nullable=False, default=0)
    seed_position = Column(Integer, nullable=True)
    match_length = Column(Integer, nullable=True)
    num_matches = Column(Integer, nullable=True)
    mismatches = Column(Integer, nullable=False, default=0)
    gaps = Column(Integer, nullable=True, default=0)
    indels = Column(Integer, nullable=True, default=0)
    complementarity = Column(Float, nullable=False, default=0.0)
    alignment_score = Column(Float, nullable=True)
    site_position = Column(Integer, nullable=True)
    site_length = Column(Integer, nullable=True)
    delta_g = Column(Float, nullable=True)  # null si no se calcula
    accessibility = Column(Float, nullable=True)
    conservation = Column(Float, nullable=True)
    mirna_expression = Column(Float, nullable=True)  # p.ej. read_count normalizado
    mrna_expression = Column(Float, nullable=True)

    # Etiqueta y predicción
    label = Column(Integer, nullable=True)  # 1 | 0 | null (unlabeled)
    label_source = Column(String(32), nullable=False, default="unlabeled")
    ml_probability = Column(Float, nullable=True)
    evidence = Column(String(255), nullable=True)
    model_version = Column(String(64), nullable=True)

    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="interactions")
    transcriptome = relationship("TranscriptomeReport", back_populates="interactions")

    __table_args__ = (
        Index("idx_ix_user_tr_pair", "user_id", "transcriptome_id", "mirna_id", "mrna_id"),
        Index("idx_ix_label_source", "user_id", "label_source"),
    )


class PipelineJobStatus(Base):
    __tablename__ = "pipeline_job_status"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    archivo = Column(String(255), index=True)
    tipo = Column(String(128), nullable=False)
    estado = Column(String(32), nullable=False)  # procesando | completado | error
    total_leidos = Column(Integer, default=0)
    validos_guardados = Column(Integer, default=0)
    descartados = Column(Integer, default=0)
    detalle_error = Column(Text, nullable=True)
    actualizado_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    user = relationship("User", back_populates="job_statuses")

    __table_args__ = (
        Index("idx_user_archivo_tipo", "user_id", "archivo", "tipo", unique=True),
    )


class MLModelRun(Base):
    """Corrida de entrenamiento reproducible."""

    __tablename__ = "ml_model_runs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    transcriptome_id = Column(
        Integer, ForeignKey("transcriptome_reports.id", ondelete="CASCADE"), index=True, nullable=False
    )
    model_version = Column(String(64), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)  # Logistic Regression | Random Forest | ...
    training_dataset = Column(String(255), nullable=True)
    features_used = Column(Text, nullable=True)  # JSON serializado
    hyperparameters = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    pr_auc = Column(Float, nullable=True)
    artifact_path = Column(String(512), nullable=True)  # .joblib en disco
    plots_dir = Column(String(512), nullable=True)
    trained_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="ml_runs")
    transcriptome = relationship("TranscriptomeReport", back_populates="ml_runs")

    __table_args__ = (
        Index("idx_ml_user_tr_model", "user_id", "transcriptome_id", "model_name", "model_version"),
    )


# ---------------------------------------------------------------------------
# Alias de compatibilidad temporal (imports antiguos durante la transición)
# ---------------------------------------------------------------------------
MirnaReport = MirnaInputSummary  # ya no guarda secuencias fila a fila
MirnaClassificationReport = IdentifiedMirna
InteractionReport = InteractionCandidate
MLModelMetricsReport = MLModelRun
