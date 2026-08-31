---
title: Academic Advisor AI Recommender
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🎓 Academic Advisor AI Recommendation Engine

Deep Learning microservice for university academic advising, smart prerequisite-aware course prioritization, and credit-budget packing.

## 🚀 Live API Endpoints

- **Interactive Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **Health Check:** `GET /health`
- **Course Recommendation:** `POST /api/v1/recommend`
- **Upload Transcript:** `POST /api/v1/upload-transcript`
- **Trigger Retraining:** `POST /api/v1/retrain`
- **Extracted Catalog:** `GET /api/v1/catalog`

## 💡 How Continuous Streaming Data Works

1. **Upload New Transcripts Anytime:**
   Send a `POST` request to `/api/v1/upload-transcript` with the new student's PDF file.
2. **Automated Incremental Retraining:**
   Send a `POST` request to `/api/v1/retrain` to automatically re-extract and fine-tune the model weights across all transcripts.
