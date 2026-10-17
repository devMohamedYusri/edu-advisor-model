import os
import torch
from src.config import MODEL_PATH, VOCAB_PATH, ARTIFACTS_DIR, EMBED_DIM
from src.utils import load_json
from src.model.network import AcademicAdvisorScorer

def export_to_onnx(model_path=MODEL_PATH, vocab_path=VOCAB_PATH, output_onnx_path=None):
    if output_onnx_path is None:
        output_onnx_path = os.path.join(ARTIFACTS_DIR, "advisor_model.onnx")

    course_to_id = load_json(vocab_path)
    if not course_to_id:
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}. Run training first.")

    model = AcademicAdvisorScorer(num_courses=len(course_to_id), embed_dim=EMBED_DIM)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()

    # Dummy inputs for ONNX tracing
    # Shape: (batch_size=1, num_passed_courses=7) and (batch_size=1, num_candidate_courses=6)
    dummy_history = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    dummy_candidates = torch.tensor([[8, 9, 10, 11, 12, 13]], dtype=torch.long)

    try:
        torch.onnx.export(
            model,
            (dummy_history, dummy_candidates),
            output_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['history_ids', 'candidate_ids'],
            output_names=['scores'],
            dynamic_axes={
                'history_ids': {0: 'batch_size', 1: 'history_length'},
                'candidate_ids': {0: 'batch_size', 1: 'num_candidates'},
                'scores': {0: 'batch_size', 1: 'num_candidates'}
            },
            dynamo=False
        )
        print(f"✅ ONNX model successfully exported to: {output_onnx_path}")
    except TypeError:
        # For older PyTorch versions without dynamo argument
        torch.onnx.export(
            model,
            (dummy_history, dummy_candidates),
            output_onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['history_ids', 'candidate_ids'],
            output_names=['scores'],
            dynamic_axes={
                'history_ids': {0: 'batch_size', 1: 'history_length'},
                'candidate_ids': {0: 'batch_size', 1: 'num_candidates'},
                'scores': {0: 'batch_size', 1: 'num_candidates'}
            }
        )
        print(f"✅ ONNX model successfully exported to: {output_onnx_path}")
    return output_onnx_path

if __name__ == "__main__":
    export_to_onnx()
