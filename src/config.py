import os
import re

DATA_DIR = os.getenv("DATA_DIR", "data")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "src/artifacts")
OUTPUT_JSON_PATH = os.getenv("OUTPUT_JSON_PATH", os.path.join(ARTIFACTS_DIR, "extracted_transcripts.json"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(ARTIFACTS_DIR, "advisor_model.pt"))
VOCAB_PATH = os.getenv("VOCAB_PATH", os.path.join(ARTIFACTS_DIR, "course_vocab.json"))

EMBED_DIM = 64
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MAX_SEMESTER_HOURS = 21
COURSE_CODE_REGEX = re.compile(r'\b([A-Za-z]{2,4})\s*(\d{3})\b')
HOURS_REGEX = re.compile(r'\b([1-4])\b\s+(?:50|100)')
YEAR_REGEX = re.compile(r'20\d\d\s*[/\\-]\s*20\d\d')

ARABIC_SEMESTER_KEYWORDS = ["فصل", "دراسي", "المستوى", "العام الأكاديمي", "الترم", "خريف", "ربيع", "صيف", "المعدل"]
REVERSED_ARABIC_KEYWORDS = ["لصف", "يسارد", "وتسم", "يميداكأ", "مرت", "لدعملا"]
ENGLISH_SEMESTER_KEYWORDS = ["semester", "term", "level", "fall", "spring", "summer", "gpa", "sgpa", "cgpa", "academic year"]
FAIL_KEYWORDS = ["X", "راسب", "غياب", " 0.00 ", " 0.00\t"]

API_TITLE = "Academic Advisor AI Recommendation Engine"
API_VERSION = "1.1.0"
DEFAULT_PORT = int(os.getenv("PORT", 7860))
