from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class FileValidationItem(BaseModel):
    role: str
    filename: str
    status: str
    format_ok: Optional[bool] = None
    size_ok: Optional[bool] = None
    structure_ok: Optional[bool] = None
    looks_like_mirna: Optional[bool] = None
    total_records: Optional[int] = None
    invalid_records: Optional[int] = None
    avg_quality: Optional[float] = None
    detail: Optional[str] = None


class JobStatusItem(BaseModel):
    archivo: Optional[str] = None
    tipo: Optional[str] = None
    estado: Optional[str] = None
    total_leidos: Optional[int] = None
    validos_guardados: Optional[int] = None
    descartados: Optional[int] = None
    error: Optional[str] = None


class MirnaSummaryItem(BaseModel):
    archivo: str
    collapsed_path: str
    total_reads: int
    reads_after_qc: int
    unique_sequences: int
    discarded: int
    min_len: Optional[int] = None
    max_len: Optional[int] = None


class TranscriptomeSummaryItem(BaseModel):
    id: int
    archivo: str
    total_transcritos: Optional[int] = None
    transcritos_validos: Optional[int] = None
    longitud_minima_bp: Optional[int] = None
    longitud_maxima_bp: Optional[int] = None
    longitud_promedio_bp: Optional[float] = None
    index_path: Optional[str] = None


class ValidationStatusResponse(BaseModel):
    status: str
    user_id: int
    mensaje: Optional[str] = None
    validaciones: List[FileValidationItem] = []
    jobs: List[JobStatusItem] = []
    mirna_summary: Optional[MirnaSummaryItem] = None
    transcriptome_summary: Optional[TranscriptomeSummaryItem] = None
    next_steps: List[str] = []


class PipelineUploadResponse(BaseModel):
    message: str
    user_id: int
    archivos_en_proceso: List[str]
    rango_mirna_aplicado: Dict[str, int]
    storage_policy: str
    mirbase_status: Dict[str, Any]
    validation_get: str
    input_summary_get: str

    model_config = ConfigDict(from_attributes=True)


class ModelMetricItem(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None


class AnalysisReportDocument(BaseModel):
    """Metadatos del informe de la búsqueda actual antes de descargar el PDF."""

    user_id: int
    transcriptome_id: int
    archivo: Optional[str] = None
    generated_at: datetime
    n_interactions: int
    n_identified_mirnas: int
    mean_complementarity: Optional[float] = None
    mean_delta_g: Optional[float] = None
    seed_match_fraction: Optional[float] = None
    model_version: Optional[str] = None
    models: List[ModelMetricItem] = []
    plots: List[str] = []
    pdf_url: str
    top_pairs: List[Dict[str, Any]] = []


class InputSummaryResponse(BaseModel):
    status: str
    user_id: int
    mensaje: Optional[str] = None
    input_files: List[Dict[str, Any]] = []
    mirna_summaries: List[MirnaSummaryItem] = []
    transcriptome_summaries: List[TranscriptomeSummaryItem] = []
    validation_get: str
