from typing import List, Optional
from pydantic import BaseModel, Field

class CourseInput(BaseModel):
    code: str = Field(..., example="CS142")
    hours: int = Field(..., ge=1, le=6, example=3)
    name: Optional[str] = Field("", example="Programming 2")

class RecommendationRequest(BaseModel):
    passed_courses: List[str] = Field(..., example=["CS141", "BS101", "BS111", "BS105", "BS103", "CS101", "UNI008"])
    available_courses: List[CourseInput]
    max_credit_hours: int = Field(..., ge=1, le=30, example=9)

class CourseRecommendation(BaseModel):
    code: str
    hours: int
    name: str
    advisor_score: float

class RecommendationResponse(BaseModel):
    success: bool = True
    total_credit_hours: int
    max_allowed_hours: int
    suggested_courses: List[CourseRecommendation]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vocabulary_tokens_count: int
    total_pdfs_in_dataset: int
    version: str

class UploadTranscriptResponse(BaseModel):
    success: bool
    filename: str
    extracted_semesters_count: int
    retrained: bool

class RetrainResponse(BaseModel):
    success: bool
    message: str
    updated_vocabulary_size: int
    model_path: str
