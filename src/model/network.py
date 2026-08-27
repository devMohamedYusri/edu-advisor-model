import torch
import torch.nn as nn

class AcademicAdvisorScorer(nn.Module):
    def __init__(self, num_courses, embed_dim=64):
        super().__init__()
        self.course_embeddings = nn.Embedding(num_courses, embed_dim, padding_idx=0)
        self.history_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        self.scoring_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, history_ids, candidate_ids):
        if history_ids.dim() == 1:
            history_ids = history_ids.unsqueeze(0)
        if candidate_ids.dim() == 1:
            candidate_ids = candidate_ids.unsqueeze(0)

        hist_embeds = self.course_embeddings(history_ids)
        mask = (history_ids != 0).unsqueeze(-1)
        hist_sum = (hist_embeds * mask).sum(dim=1)
        hist_len = mask.sum(dim=1).clamp(min=1)
        student_vec = self.history_fc(hist_sum / hist_len)

        cand_embeds = self.course_embeddings(candidate_ids)
        _, num_cand, _ = cand_embeds.shape
        student_vec_expanded = student_vec.unsqueeze(1).expand(-1, num_cand, -1)
        combined = torch.cat([student_vec_expanded, cand_embeds], dim=-1)

        logits = self.scoring_head(combined).squeeze(-1)
        return torch.sigmoid(logits)
