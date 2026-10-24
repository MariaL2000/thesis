# Pipeline miRNA–mRNA (sequía) — API FastAPI

Sistema para identificar y priorizar posibles interacciones miRNA–mRNA a partir de small RNA-seq y un transcriptoma, con QC, comparación contra miRBase (`mature.fa`), extracción de características y Machine Learning.

SQLite en local; PostgreSQL en la nube más adelante. Por límites de almacenamiento (≈500 MB–1 GB), **las secuencias viven en disco** y la DB solo guarda metadatos, matches e interacciones.

---

## Qué se hizo (rectificación)

### Política de almacenamiento

| En disco (`data/`) | En DB (SQLite / futuro PostgreSQL) |
|---|---|
| FASTA/FASTQ crudos (`Secuencias/user_{id}/raw/`) | `User`, `InputFile`, `ValidationReport`, jobs |
| Lecturas colapsadas `seq → count` (`.tsv`) | `MirnaInputSummary` (ruta + conteos) |
| Índice de transcritos (`.tsv`) | `TranscriptRecord` (id, gene, length; **sin secuencia**) |
| `mature.fa` + modelos / plots | `IdentifiedMirna`, `InteractionCandidate`, `MLModelRun` |

### Pipeline

1. **Upload** — 2 entradas obligatorias (miRNA + transcriptoma), 1 archivo c/u, máx 1 GB, formatos FASTA / FASTQ / GZ.
2. **Validación** — peek estructural (~200 lecturas) + QC completo en background; consultable por GET.
3. **QC miRNA** — adaptadores Illumina, filtro 18–25 nt (configurable), colapsado a disco.
4. **miRBase** — índice hash de `mature.fa`; en DB solo matches + top-100 novel.
5. **Interacciones** — pares candidatos + features; negativas sintéticas (`label_source=synthetic_negative`); resto `unlabeled`.
6. **ML** — Logistic Regression + Random Forest sobre filas etiquetadas; métricas en `MLModelRun`; plots con Matplotlib.

### Cambios importantes respecto al código anterior

- Ya **no** se guardan millones de lecturas en tablas fila a fila.
- Ya **no** se trunca el input con `max_records`.
- El ML **no** usa etiquetas aleatorias (`np.random`).
- Cada etapa tiene GET para ver progreso/resultados.

---

## Requisitos

- Python 3.10+ recomendado  
- Dependencias en `requirements.txt`

```powershell
cd D:\Trabajo\Lesyani\thesis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## Arranque

Tras cambios de esquema, borra la DB local y levanta la API:

```powershell
Remove-Item .\analisis_bio.db -ErrorAction SilentlyContinue
python -m uvicorn app.main:app --reload
```

- API: http://127.0.0.1:8000  
- Swagger: http://127.0.0.1:8000/docs  
- Health: `GET /` → mensaje de OK  

Al iniciar, `create_all` crea las tablas nuevas.

---

## Fixtures de demo

Incluidas en el repo:

| Archivo | Uso |
|---|---|
| `data/fixtures/demo_mirna.fa` | Entrada 1 (miRNA / sRNA) |
| `data/fixtures/demo_transcriptome.fa` | Entrada 2 (mRNA) |
| `data/Secuencias/mature.fa` | Referencia miRBase mínima (arroz/Arabidopsis) |

Sustituye `mature.fa` por el completo de miRBase cuando tengas el archivo real.

---

## Cómo probar todo (orden recomendado)

Usa Swagger o `curl` / Postman. En cada POST anota el `user_id` (si envías `0`, se crea uno nuevo).

### 1) Subir entradas

`POST /api/v1/upload-inputs/`

- Form:
  - `user_id` = `0` (primera vez)
  - `min_len_mirna` = `18`
  - `max_len_mirna` = `25`
  - `mirna_file` = `data/fixtures/demo_mirna.fa`
  - `transcriptome_file` = `data/fixtures/demo_transcriptome.fa`

Respuesta útil:

- `user_id`
- `validation_get` → ruta del GET de validación
- `input_summary_get`

Ejemplo con curl (PowerShell):

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/upload-inputs/" `
  -F "user_id=0" `
  -F "min_len_mirna=18" `
  -F "max_len_mirna=25" `
  -F "mirna_file=@data/fixtures/demo_mirna.fa" `
  -F "transcriptome_file=@data/fixtures/demo_transcriptome.fa"
```

### 2) Ver validación / QC (polling)

`GET /api/v1/validation-status/{user_id}`

Estados típicos: `procesando` → `completado` (o `error` / `parcial`).

Cuando `status` sea `completado` verás:

- `validaciones` (format/size/structure, `looks_like_mirna`, calidad)
- `mirna_summary` (únicos, descartados, ruta del TSV colapsado)
- `transcriptome_summary`
- `next_steps`

También: `GET /api/v1/input-summary/{user_id}`

### 3) Comparar con miRBase

Asegúrate de tener `data/Secuencias/mature.fa` (copia completa de miRBase). El parser lee cabeceras del tipo:

`cel-let-7-5p MIMAT0000001 Caenorhabditis elegans let-7-5p`

y rellena `mirbase_id` + `species` (nombre completo) en matches **exact** y **seed**.

`POST /api/v1/mirbase/comparar-guardados/`

- Form: `user_id` = el de tu sesión

Luego: `GET /api/v1/mirbase/estadisticas-clasificacion/{user_id}`

Espera `status: completado` y revisa `tabla_muestra_resultados` (`mirbase_id`, `species`, `match_type`).

### 4) Interacciones (si se quedaba cargando)

El sitio en el mRNA se busca como **complemento reverso del seed** (no la misma secuencia). Además hay tope de miRNAs/transcritos para que el job termine.

`POST /api/v1/interaction/calcular-interacciones` → `GET /api/v1/interaction/estadisticas-interaccion/{user_id}`

Si `status` es `error`, mira `mensaje` / `detalle_error` del job.

### 5) Machine Learning

Regresión logística y Random Forest se comparan porque el dataset es tabular y de tamaño moderado: la logística muestra el peso lineal de cada feature (coeficientes) y el bosque captura relaciones no lineales (importancia Gini). Cada entrenamiento reutiliza los pares ya etiquetados de ese usuario.

`POST /api/v1/ml/entrenar-interacciones/{user_id}`

Luego: `GET /api/v1/ml/metricas/{user_id}`

Gráficos en `data/Visuales/user_{id}_run_{transcriptome_id}/`:
ROC, precision-recall, importancia, histograma de complementariedad, ΔG vs complementariedad y conteo de seed match.

Hay que volver a calcular interacciones para que ΔG y complementariedad salgan del dúplex (Biopython + NumPy), no de la fórmula anterior.

### 6) PDF de la búsqueda actual

`GET /api/v1/report/{user_id}` devuelve el schema (conteos, métricas, pares top, lista de gráficos).

`GET /api/v1/report/{user_id}/pdf` descarga el PDF de esa búsqueda.

La variante con nivel de evidencia es `POST /api/v1/ml/pipeline-completo/{user_id}` (usa el mismo entrenamiento y añade la etiqueta de evidencia).

---

## Estructura del proyecto

```
app/
  main.py
  core/config.py          # rutas, límites, labels
  models/models.py        # esquema DB mínimo
  schemas/schemas.py      # respuestas Pydantic
  api/v1/endpoints/
    analysis.py           # upload + validation GET + input-summary
    mirbase_analysis.py
    interaction_analysis.py
    mml_interaction.py
    validation.py         # ML + plots
  services/
    inputs_service.py     # upload, peek, QC, colapsado
    disk_store.py         # TSV colapsado / índice
    mirbase_service.py
    interaction_service.py
    mml_service.py
    binding_affinity.py   # complementariedad y ΔG
    plots_service.py
    report_service.py
    validation.py
    database.py
data/
  fixtures/               # demos
  Secuencias/             # mature.fa + user_{id}/raw|processed
  Visuales/               # curvas ROC / importancia
```

---

## Notas para la tesis / producción

- **Rango de longitud**: default 18–25 nt; opcional hasta 30 para plantas. Evitar 40–50 como default.
- **Etiquetas ML**: no marcar como negativas todos los pares sin evidencia. Hoy: negativas sintéticas (sin seed) + unlabeled; positivas de evidencia se pueden añadir después (`label_source=evidence`).
- **ΔG / accesibilidad**: placeholders documentados hasta integrar herramienta real (p. ej. ViennaRNA).
- **Cloud**: migrar `DATABASE_URL` en `database.py` a PostgreSQL; mantener secuencias en disco/objeto storage, no en la DB.

---

## Checklist rápido

- [ ] venv + `pip install -r requirements.txt`
- [ ] Borrar `analisis_bio.db` tras cambios de modelos
- [ ] `uvicorn` + `/docs`
- [ ] Upload con fixtures
- [ ] GET validation-status → `completado`
- [ ] miRBase → estadísticas
- [ ] Interacciones → estadísticas
- [ ] ML → métricas (y plots si usas pipeline-completo)
