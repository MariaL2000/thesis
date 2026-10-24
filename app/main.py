from app.api.v1.endpoints import analysis, mirbase_analysis, interaction_analysis, mml_interaction, report, validation
from app.models.models import (
    Base,  # noqa: F401
)
from app.services.database import engine
from fastapi import FastAPI

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Pipeline Bioinformático miRNA",
    version="1.0",
    description="API para análisis de interacción miRNA-mRNA y Machine Learning",
)

app.include_router(analysis.router, prefix="/api/v1", tags=["Análisis Estadístico y Visual"])
app.include_router(mirbase_analysis.router, prefix="/api/v1", tags=["Comparación miRBase"])
app.include_router(interaction_analysis.router, prefix="/api/v1", tags=["Predicción y Características de Interacción"])
app.include_router(mml_interaction.router, prefix="/api/v1", tags=["Machine Learning e Interacciones"])
app.include_router(validation.router,prefix="/api/v1", tags=["Novelty sore y validaciones de resultados"])
app.include_router(report.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "API de análisis de miRNA funcionando correctamente"}