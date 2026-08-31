import torch
import torch.nn as nn
import torch.optim as optim
from src.config import (
    OUTPUT_JSON_PATH, MODEL_PATH, VOCAB_PATH,
    EMBED_DIM, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
)
from src.utils import save_json, build_vocab
from src.model.network import AcademicAdvisorScorer
from src.data.processor import load_training_dataset, process_all_transcripts

def prepare_tensors(samples, course_to_id):
    if not samples:
        return torch.zeros((1, 1), dtype=torch.long), torch.zeros((1, 1), dtype=torch.long), torch.zeros((1,), dtype=torch.float32)

    max_hist_len = max(len(s.get("student_history", [])) for s in samples)
    max_hist_len = max(max_hist_len, 1)
    
    history_ids_list = []
    candidate_ids_list = []
    target_scores_list = []

    for s in samples:
        hist = s.get("student_history", [])
        hist_encoded = [course_to_id.get(c, 0) for c in hist] or [0]
        padded_hist = hist_encoded + [0] * (max_hist_len - len(hist_encoded))
        
        cand_encoded = course_to_id.get(s.get("candidate_id"), 0)
        
        history_ids_list.append(padded_hist)
        candidate_ids_list.append([cand_encoded])
        target_scores_list.append(float(s.get("target_score", 0.0)))

    return (
        torch.tensor(history_ids_list, dtype=torch.long),
        torch.tensor(candidate_ids_list, dtype=torch.long),
        torch.tensor(target_scores_list, dtype=torch.float32)
    )

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func=None, duration=None):
            def decorator(f):
                return f
            if func is not None and callable(func):
                return func
            return decorator

@spaces.GPU
def train_and_export(force_reprocess=True, epochs=EPOCHS, batch_size=BATCH_SIZE, model_path=MODEL_PATH, vocab_path=VOCAB_PATH):
    dataset = load_training_dataset(OUTPUT_JSON_PATH, force_reprocess=force_reprocess)
    samples = dataset.get("training_samples", [])
    
    if not samples:
        dataset = process_all_transcripts()
        samples = dataset.get("training_samples", [])

    course_to_id = build_vocab(dataset)
    save_json(course_to_id, vocab_path)

    history_ids, candidate_ids, targets = prepare_tensors(samples, course_to_id)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AcademicAdvisorScorer(num_courses=len(course_to_id), embed_dim=EMBED_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()

    model.train()
    dataset_tensor = torch.utils.data.TensorDataset(history_ids, candidate_ids, targets)
    actual_batch_size = min(batch_size, len(samples)) if samples else 1
    dataloader = torch.utils.data.DataLoader(dataset_tensor, batch_size=actual_batch_size, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_hist, batch_cand, batch_targets in dataloader:
            batch_hist = batch_hist.to(device)
            batch_cand = batch_cand.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            predictions = model(batch_hist, batch_cand).squeeze(-1)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_targets)

        if epoch % 30 == 0 or epoch == epochs:
            avg_loss = epoch_loss / len(samples) if samples else 0.0
            print(f"Epoch [{epoch:3d}/{epochs}] - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Saved weights to '{model_path}' and vocab to '{vocab_path}'.")
    return model_path

if __name__ == "__main__":
    train_and_export(force_reprocess=True)
