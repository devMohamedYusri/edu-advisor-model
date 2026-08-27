import os
import json
from src.config import MODEL_PATH, VOCAB_PATH, ARTIFACTS_DIR

def load_json(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def check_artifacts_exist(model_path=MODEL_PATH, vocab_path=VOCAB_PATH):
    return os.path.exists(model_path) and os.path.exists(vocab_path)

def build_vocab(dataset):
    vocab = {"<PAD>": 0}
    catalog = dataset.get("course_catalog", {})
    for code in catalog.keys():
        if code not in vocab:
            vocab[code] = len(vocab)
            
    for sample in dataset.get("training_samples", []):
        for c in sample.get("student_history", []):
            if c not in vocab:
                vocab[c] = len(vocab)
        cand = sample.get("candidate_id")
        if cand and cand not in vocab:
            vocab[cand] = len(vocab)
            
    return vocab

def pack_credit_bundle(ranked_courses, max_credit_hours):
    selected_bundle = []
    accumulated_hours = 0
    for course in ranked_courses:
        if accumulated_hours + course['hours'] <= max_credit_hours:
            selected_bundle.append(course)
            accumulated_hours += course['hours']
            if accumulated_hours == max_credit_hours:
                break
    return {
        "total_credit_hours": accumulated_hours,
        "max_allowed_hours": max_credit_hours,
        "suggested_courses": selected_bundle
    }
