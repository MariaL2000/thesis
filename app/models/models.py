from datetime import datetime, timezone
from app.services.database import Base
from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey, Index
from sqlalchemy.orm import relationship


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    transcriptomes = relationship("TranscriptomeReport", back_populates="user", cascade="all, delete-orphan")
    mirnas = relationship("MirnaReport", back_populates="user", cascade="all, delete-orphan")
    classifications = relationship("MirnaClassificationReport", back_populates="user", cascade="all, delete-orphan")
    interactions = relationship("InteractionReport", back_populates="user", cascade="all, delete-orphan")
    job_statuses = relationship("PipelineJobStatus", back_populates="user", cascade="all, delete-orphan")
    
    # ➕ FALTANTE AÑADIDO: Relación bidireccional con las métricas del modelo de ML
    ml_metrics = relationship("MLModelMetricsReport", back_populates="user", cascade="all, delete-orphan")


class TranscriptomeReport(Base):
    __tablename__ = "transcriptome_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    fecha = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    archivo = Column(String, index=True)
    total_transcritos = Column(Integer)
    transcritos_validos = Column(Integer)
    registros_invalidos_descartados = Column(Integer)
    longitud_minima_bp = Column(Integer)
    longitud_maxima_bp = Column(Integer)
    longitud_promedio_bp = Column(Float)
    longitud_mediana_bp = Column(Float)
    top_prefijo = Column(String)
    top_prefijo_conteo = Column(Integer)

    user = relationship("User", back_populates="transcriptomes")
    interactions = relationship("InteractionReport", back_populates="transcriptome", cascade="all, delete-orphan")
    
    # ➕ FALTANTE AÑADIDO: Relación bidireccional con las métricas de ML por transcriptoma
    ml_metrics = relationship("MLModelMetricsReport", back_populates="transcriptome", cascade="all, delete-orphan")


class MirnaReport(Base):
    __tablename__ = "mirna_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    fecha = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    archivo = Column(String, index=True)
    miRNA = Column(String, index=True)      # ID o nombre del miRNA
    secuencia = Column(String)              # Secuencia limpia validada (18-25 nt)
    longitud = Column(Integer)              # Longitud exacta para filtros rápidos

    user = relationship("User", back_populates="mirnas")


class MirnaClassificationReport(Base):
    __tablename__ = "mirna_classification_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    archivo = Column(String, index=True)
    miRNA = Column(String, index=True)
    secuencia = Column(String)
    clasificacion = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="classifications")
    

class InteractionReport(Base):
    __tablename__ = "interaction_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    transcriptome_id = Column(Integer, ForeignKey("transcriptome_reports.id", ondelete="CASCADE"), index=True, nullable=False)
    
    miRNA = Column(String(100), index=True, nullable=False)
    mRNA = Column(String(100), index=True, nullable=False)
    
    # Características para Machine Learning
    complementariedad = Column(Float, nullable=False)
    seed_match = Column(Integer, nullable=False)
    mismatches = Column(Integer, nullable=False)
    delta_g = Column(Float, nullable=False)
    accesibilidad = Column(Float, nullable=False)
    interaction_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="interactions")
    transcriptome = relationship("TranscriptomeReport", back_populates="interactions")

    # Índice compuesto optimizado por usuario, transcriptoma y pares
    __table_args__ = (
        Index('idx_user_transcriptome_mirna_mrna', 'user_id', 'transcriptome_id', 'miRNA', 'mRNA'),
    )


class PipelineJobStatus(Base):
    __tablename__ = "pipeline_job_status"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    archivo = Column(String, index=True)  # Ajustado para permitir nombres repetidos entre diferentes usuarios
    tipo = Column(String)
    estado = Column(String)  # "procesando", "completado", "error"
    total_leidos = Column(Integer, default=0)
    validos_guardados = Column(Integer, default=0)
    descartados = Column(Integer, default=0)
    detalle_error = Column(String, nullable=True)
    actualizado_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="job_statuses")

    # Asegura unicidad del estado del job por usuario, archivo y tipo
    __table_args__ = (
        Index('idx_user_archivo_tipo', 'user_id', 'archivo', 'tipo', unique=True),
    )


class MLModelMetricsReport(Base):
    __tablename__ = "ml_model_metrics_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    transcriptome_id = Column(Integer, ForeignKey("transcriptome_reports.id", ondelete="CASCADE"), index=True, nullable=False)
    
    modelo = Column(String(100), index=True, nullable=False)  # Ej: Logistic Regression, Random Forest, XGBoost
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    pr_auc = Column(Float, nullable=False)
    
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="ml_metrics")
    transcriptome = relationship("TranscriptomeReport", back_populates="ml_metrics")

    __table_args__ = (
        Index('idx_user_transcriptome_modelo', 'user_id', 'transcriptome_id', 'modelo', unique=True),
    )