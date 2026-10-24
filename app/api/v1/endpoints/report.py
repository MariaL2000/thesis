from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.schemas import AnalysisReportDocument
from app.services.report_service import report_document, write_user_pdf

router = APIRouter(prefix="/report", tags=["Informe PDF"])


@router.get("/{user_id}", response_model=AnalysisReportDocument)
def obtener_informe(user_id: int):
    """Schema del informe de la búsqueda más reciente de este usuario."""
    try:
        return report_document(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{user_id}/pdf")
def descargar_informe_pdf(user_id: int):
    """PDF de la búsqueda actual: metodología, métricas, pares y gráficos."""
    try:
        path = write_user_pdf(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name,
    )
