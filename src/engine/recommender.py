import json
import torch
from src.config import MODEL_PATH, VOCAB_PATH, EMBED_DIM
from src.utils import load_json, check_artifacts_exist, pack_credit_bundle
from src.model.network import AcademicAdvisorScorer

def ensure_model_and_vocab_exist(model_path=MODEL_PATH, vocab_path=VOCAB_PATH):
    if not check_artifacts_exist(model_path, vocab_path):
        from src.engine.trainer import train_and_export
        train_and_export(model_path=model_path, vocab_path=vocab_path)

def recommend_registration_bundle(passed_courses, available_courses, max_credit_hours, model_path=MODEL_PATH, vocab_path=VOCAB_PATH):
    ensure_model_and_vocab_exist(model_path=model_path, vocab_path=vocab_path)

    course_to_id = load_json(vocab_path)
    model = AcademicAdvisorScorer(num_courses=len(course_to_id), embed_dim=EMBED_DIM)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    hist_tokens = [course_to_id.get(c, 0) for c in passed_courses] or [0]
    cand_tokens = [course_to_id.get(c['code'], 0) for c in available_courses]

    hist_ids = torch.tensor([hist_tokens], dtype=torch.long)
    cand_ids = torch.tensor([cand_tokens], dtype=torch.long)

    with torch.no_grad():
        raw_scores = model(hist_ids, cand_ids).squeeze(0)
        scores = [raw_scores.item()] if raw_scores.ndim == 0 else raw_scores.tolist()

    evaluated_candidates = []
    for i, c in enumerate(available_courses):
        item = dict(c)
        item['advisor_score'] = round(float(scores[i]), 4)
        evaluated_candidates.append(item)

    ranked_candidates = sorted(evaluated_candidates, key=lambda x: x['advisor_score'], reverse=True)
    return pack_credit_bundle(ranked_candidates, max_credit_hours)

if __name__ == "__main__":
    passed = ['CS141', 'BS101', 'BS111', 'BS105', 'BS103', 'CS101', 'UNI008']
    candidates = [
        {'code': 'CS142', 'hours': 3, 'name': 'Programming 2'},
        {'code': 'CS102', 'hours': 3, 'name': 'Logic Design'},
        {'code': 'BS102', 'hours': 3, 'name': 'Mathematics 2'},
        {'code': 'BS133', 'hours': 2, 'name': 'Computing Economics'},
        {'code': 'UNI001', 'hours': 1, 'name': 'Human Rights'},
        {'code': 'UNI003', 'hours': 1, 'name': 'Problem Solving'}
    ]
    recommendation = recommend_registration_bundle(passed, candidates, 9)
    print(json.dumps(recommendation, indent=2))
