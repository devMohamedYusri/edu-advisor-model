import os
import glob
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from src.config import (
    API_TITLE, API_VERSION, DEFAULT_PORT,
    DATA_DIR, OUTPUT_JSON_PATH, MODEL_PATH, VOCAB_PATH
)
from src.schemas import (
    RecommendationRequest, RecommendationResponse,
    HealthResponse, UploadTranscriptResponse, RetrainResponse
)
from src.utils import load_json
from src.engine.recommender import recommend_registration_bundle, ensure_model_and_vocab_exist
from src.data.processor import parse_pdf_transcript
from src.engine.trainer import train_and_export

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    ensure_model_and_vocab_exist()

@app.get("/api/v1/info", tags=["System"])
def api_info():
    return {
        "service": API_TITLE,
        "status": "online",
        "docs": "/docs",
        "endpoints": {
            "recommend": "POST /api/v1/recommend",
            "upload_transcript": "POST /api/v1/upload-transcript",
            "retrain": "POST /api/v1/retrain",
            "catalog": "GET /api/v1/catalog",
            "health": "GET /health"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    vocab_data = load_json(VOCAB_PATH) or {}
    pdf_count = len(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    
    return {
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "vocabulary_tokens_count": len(vocab_data),
        "total_pdfs_in_dataset": pdf_count,
        "version": API_VERSION
    }

@app.post("/api/v1/recommend", response_model=RecommendationResponse, tags=["Advisor AI"])
def get_recommendation(payload: RecommendationRequest):
    if not payload.available_courses:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="available_courses list cannot be empty."
        )

    try:
        candidates_data = [c.dict() for c in payload.available_courses]
        result = recommend_registration_bundle(
            passed_courses=payload.passed_courses,
            available_courses=candidates_data,
            max_credit_hours=payload.max_credit_hours
        )
        return {
            "success": True,
            "total_credit_hours": result["total_credit_hours"],
            "max_allowed_hours": result["max_allowed_hours"],
            "suggested_courses": result["suggested_courses"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/api/v1/upload-transcript", response_model=UploadTranscriptResponse, tags=["Data Streaming"])
async def upload_transcript(file: UploadFile = File(...), auto_retrain: bool = False):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF transcript files are accepted."
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, file.filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parsed = parse_pdf_transcript(save_path)
    
    if auto_retrain:
        train_and_export(force_reprocess=True)
        
    return {
        "success": True,
        "filename": file.filename,
        "extracted_semesters_count": parsed["total_semesters"],
        "retrained": auto_retrain
    }

@app.post("/api/v1/retrain", response_model=RetrainResponse, tags=["Data Streaming"])
def trigger_retraining(epochs: int = 150):
    try:
        train_and_export(force_reprocess=True, epochs=epochs)
        vocab_data = load_json(VOCAB_PATH) or {}
        return {
            "success": True,
            "message": "Model training completed successfully.",
            "updated_vocabulary_size": len(vocab_data),
            "model_path": MODEL_PATH
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/v1/catalog", tags=["Advisor AI"])
def get_course_catalog():
    data = load_json(OUTPUT_JSON_PATH)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Catalog not found. Run training first.")
        
    return {
        "unique_courses_count": len(data.get("course_catalog", {})),
        "catalog": data.get("course_catalog", {})
    }
