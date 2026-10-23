"""FastAPI REST API server managing model lifecycle with async lifespan and static UI mounting."""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import yaml

from src.inference import (
    CorruptImageError,
    GlaucomaInferenceEngine,
    InferenceError,
    MetadataContractError,
    ModelCheckpointError,
)
from src.report import generate_clinical_pdf

logger = logging.getLogger(__name__)

# Global singleton engine instance initialized during lifespan
_engine: Optional[GlaucomaInferenceEngine] = None


def get_engine() -> GlaucomaInferenceEngine:
    """Retrieve initialized inference engine singleton."""
    global _engine
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GlaucomaInferenceEngine is not initialized or still starting up.",
        )
    return _engine


# ==============================================================================
# Pydantic v2 Response Schemas
# ==============================================================================

class QualityWarning(BaseModel):
    message: str


class PredictionResponse(BaseModel):
    reference_id: str = Field(..., description="Unique request tracking identifier")
    predicted_class: str = Field(..., description="Classification outcome: 'Glaucoma' or 'Normal'")
    suspicion_score: float = Field(
        ...,
        description="Continuous Glaucoma Suspicion Score in [0, 1]. Not a calibrated probability.",
    )
    threshold_used: float = Field(..., description="Frozen decision threshold from validation calibration")
    threshold_procedure: str = Field(..., description="Validation calibration procedure name")
    latency_ms: float = Field(..., description="Total in-memory processing latency in milliseconds")
    quality_warnings: List[str] = Field(
        default_factory=list,
        description="Non-blocking image quality heuristic warnings",
    )
    gradcam_overlay_base64: str = Field(
        ...,
        description="Base64-encoded PNG image of the Grad-CAM layer4[-1] overlay",
    )
    model_version: str = Field(..., description="Model architecture and weight version identifier")
    disclaimer: str = Field(..., description="Mandatory regulatory prototype disclaimer")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    device: str = Field(..., examples=["cpu"])
    threads: int = Field(..., examples=[14])
    threshold: float = Field(..., examples=[0.2921])
    threshold_procedure: str = Field(..., examples=["youden_j_on_val"])
    model_version: str = Field(..., examples=["resnet18-clean"])
    disclaimer: str = Field(..., description="Mandatory prototype disclaimer")


# ==============================================================================
# Lifespan Management
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    global _engine
    config_path = Path("configs/model_config.yaml")
    logger.info("Initializing GlaucomaInferenceEngine during FastAPI lifespan startup...")
    try:
        _engine = GlaucomaInferenceEngine(config_path=config_path)
        logger.info("GlaucomaInferenceEngine loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize GlaucomaInferenceEngine: {e}")
        _engine = None

    yield

    logger.info("Shutting down FastAPI service. Releasing resources...")
    _engine = None


# ==============================================================================
# FastAPI Application Configuration
# ==============================================================================

app = FastAPI(
    title="REFUGE Glaucoma Binary Screening API",
    version="1.0.0",
    description=(
        "Production REST API for automated binary glaucoma screening from digital fundus photographs. "
        "Strictly emits a 'Glaucoma Suspicion Score' — not a calibrated probability."
    ),
    lifespan=lifespan,
)

# Enable CORS for local web clients and dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["System"])
@app.api_route("/api/v1/health", methods=["GET", "HEAD"], response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Verify system health, loaded model architecture, CPU threads, and decision threshold."""
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        device=str(engine.device),
        threads=engine.threads,
        threshold=engine.threshold,
        threshold_procedure=engine.threshold_procedure,
        model_version=engine.model_version,
        disclaimer=engine.disclaimer,
    )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_glaucoma(
    file: UploadFile = File(..., description="Digital fundus photograph (.jpg, .jpeg, .png)"),
) -> PredictionResponse:
    """Run automated glaucoma screening on an in-memory uploaded fundus photograph.

    - Memory-only processing: Never saves patient images to persistent storage.
    - Applies non-blocking blur and illumination quality checks.
    - Computes Glaucoma Suspicion Score and generates layer4[-1] Grad-CAM overlay.
    """
    engine = get_engine()

    # Validate file extension
    server_cfg = engine.config.get("server", {})
    allowed_exts = set(server_cfg.get("allowed_extensions", [".jpg", ".jpeg", ".png"]))
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed formats: {sorted(allowed_exts)}",
        )

    # Validate payload size limit in memory
    max_bytes = int(server_cfg.get("max_upload_bytes", 15728640))  # 15 MB default
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file bytes: {e}",
        ) from e

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({len(content)} bytes) exceeds the maximum allowed limit of {max_bytes} bytes.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )

    # Perform inference in memory
    try:
        result = engine.predict(content, include_gradcam=True)
    except CorruptImageError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable image: {e}",
        ) from e
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {e}",
        ) from e

    return PredictionResponse(
        reference_id=result["reference_id"],
        predicted_class=result["predicted_class"],
        suspicion_score=result["suspicion_score"],
        threshold_used=result["threshold_used"],
        threshold_procedure=result["threshold_procedure"],
        latency_ms=result["latency_ms"],
        quality_warnings=result["quality_warnings"],
        gradcam_overlay_base64=result["gradcam_overlay_base64"],
        model_version=result["model_version"],
        disclaimer=result["disclaimer"],
    )


@app.post(
    "/api/v1/report",
    status_code=status.HTTP_200_OK,
    tags=["Reporting"],
)
async def generate_report_pdf(
    file: UploadFile = File(..., description="Digital fundus photograph (.jpg, .jpeg, .png)"),
) -> StreamingResponse:
    """Generate in-memory single-page clinical screening PDF report for uploaded fundus image.

    - Memory-only processing: Builds PDF in io.BytesIO without writing to server disk.
    - Zero PHI: De-identified with request UUID and UTC timestamp.
    - Streams application/pdf directly back to caller.
    """
    engine = get_engine()

    server_cfg = engine.config.get("server", {})
    allowed_exts = set(server_cfg.get("allowed_extensions", [".jpg", ".jpeg", ".png"]))
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{file_ext}'. Allowed formats: {sorted(allowed_exts)}",
        )

    max_bytes = int(server_cfg.get("max_upload_bytes", 15728640))
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file bytes: {e}",
        ) from e

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({len(content)} bytes) exceeds the maximum allowed limit of {max_bytes} bytes.",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )

    try:
        result = engine.predict(content, include_gradcam=True)
        pdf_buf = generate_clinical_pdf(prediction_result=result, original_image=content)
    except CorruptImageError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Corrupt or unreadable image: {e}",
        ) from e
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure during report generation: {e}",
        ) from e

    filename = f"glaucoma_screening_report_{result['reference_id']}.pdf"
    return StreamingResponse(
        pdf_buf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=\"{filename}\"",
        },
    )


@app.get("/api/v1/evaluation", tags=["Evaluation"])
async def get_evaluation_metrics() -> JSONResponse:
    """Retrieve official held-out test evaluation report and bootstrap confidence intervals."""
    eval_path = Path("artifacts/reports/evaluation.json").resolve()
    if not eval_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation report not found. Run python -m src.evaluate first.",
        )
    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


# ==============================================================================
# Static UI Mount (Vanilla HTML5 / CSS / JS)
# ==============================================================================

static_dir = Path("src/static").resolve()
static_dir.mkdir(parents=True, exist_ok=True)

@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the root screening web interface."""
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "Frontend index.html not yet installed in src/static/."},
        )
    return FileResponse(index_file)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
