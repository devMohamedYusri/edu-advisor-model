import os
import json
import gradio as gr
from src.engine.recommender import recommend_registration_bundle
from src.data.processor import load_training_dataset, parse_pdf_transcript
from src.engine.trainer import train_and_export
from src.config import OUTPUT_JSON_PATH, DATA_DIR

try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

def spaces_wrapper(fn):
    if has_spaces:
        return spaces.GPU(fn)
    return fn

@spaces_wrapper
def gradio_recommend(passed_courses_str, max_credit_hours):
    if not passed_courses_str or not passed_courses_str.strip():
        return json.dumps({"error": "Please provide passed course codes."}, indent=2)

    passed = [c.strip().upper() for c in passed_courses_str.split(",") if c.strip()]
    dataset = load_training_dataset(OUTPUT_JSON_PATH) or {}
    catalog = dataset.get("course_catalog", {})
    available = []
    for code, info in catalog.items():
        if code not in passed:
            available.append({
                "code": code,
                "hours": info.get("hours", 3),
                "name": code
            })

    if not available:
        return json.dumps({"error": "No remaining candidate courses found in catalog."}, indent=2)

    result = recommend_registration_bundle(passed, available, int(max_credit_hours))
    return json.dumps(result, indent=2, ensure_ascii=False)

@spaces_wrapper
def gradio_upload_and_retrain(pdf_file):
    if pdf_file is None:
        return "Please upload a valid PDF transcript file."
    
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = os.path.basename(pdf_file.name)
    destination = os.path.join(DATA_DIR, filename)
    
    with open(pdf_file.name, "rb") as src, open(destination, "wb") as dst:
        dst.write(src.read())
        
    parsed = parse_pdf_transcript(destination)
    train_and_export(force_reprocess=True, epochs=150)
    
    return f"Successfully processed '{filename}' ({parsed['total_semesters']} semesters detected) and retrained model weights across the updated dataset!"

API_DOCS_MARKDOWN = """
# 📖 Academic Advisor AI — API Integration Guide

This microservice provides automated course recommendation, prerequisite prioritization, and credit-hour packing.

---

### 🌐 Live Production Base URL
`https://mohamedyusri-edu-advisor-recommendation-system.hf.space`

---

### 1. 🎯 Main Recommendation Endpoint

- **URL:** `POST https://mohamedyusri-edu-advisor-recommendation-system.hf.space/gradio_api/call/recommend`
- **Headers:** `Content-Type: application/json`

#### 📥 Request Body (JSON):
```json
{
  "data": [
    "CS141, BS101, BS111, BS105, BS103, CS101, UNI008",
    9
  ]
}
```

#### 📤 Response Body (JSON):
```json
{
  "total_credit_hours": 7,
  "max_allowed_hours": 9,
  "suggested_courses": [
    {
      "code": "UNI001",
      "hours": 1,
      "name": "Human Rights",
      "advisor_score": 1.0
    },
    {
      "code": "UNI003",
      "hours": 1,
      "name": "Problem Solving",
      "advisor_score": 1.0
    },
    {
      "code": "BS102",
      "hours": 3,
      "name": "Mathematics 2",
      "advisor_score": 0.9617
    },
    {
      "code": "BS133",
      "hours": 2,
      "name": "Computing Economics",
      "advisor_score": 0.9329
    }
  ]
}
```

---

### 2. 💻 Ready-to-Use C# .NET Example

```csharp
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

public class AcademicAdvisorClient
{
    private readonly HttpClient _httpClient;
    private const string ApiUrl = "https://mohamedyusri-edu-advisor-recommendation-system.hf.space/gradio_api/call/recommend";

    public AcademicAdvisorClient(HttpClient httpClient)
    {
        _httpClient = httpClient;
    }

    public async Task<string> GetRecommendationsAsync(string passedCoursesCsv, int maxCreditHours)
    {
        var payload = new
        {
            data = new object[] { passedCoursesCsv, maxCreditHours }
        };

        var jsonContent = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync(ApiUrl, jsonContent);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadAsStringAsync();
    }
}
```
"""

with gr.Blocks(title="Academic Advisor AI") as demo:
    gr.Markdown("# 🎓 Academic Advisor AI Recommendation Engine")

    with gr.Tabs():
        with gr.TabItem("🚀 Live Recommendation Simulator"):
            with gr.Row():
                with gr.Column():
                    passed_input = gr.Textbox(
                        label="Student Passed Course Codes (comma-separated)",
                        value="CS141, BS101, BS111, BS105, BS103, CS101, UNI008",
                        placeholder="e.g. CS141, BS101, BS111"
                    )
                    hours_slider = gr.Slider(minimum=1, maximum=24, value=9, step=1, label="Max Credit Hours Limit")
                    btn = gr.Button("Calculate Recommendation Bundle", variant="primary")

                with gr.Column():
                    output_json = gr.Code(label="Recommended Registration Bundle (JSON Output)", language="json")

            btn.click(
                gradio_recommend,
                inputs=[passed_input, hours_slider],
                outputs=output_json,
                api_name="recommend"
            )

        with gr.TabItem("📖 .NET API Documentation & Code Examples"):
            gr.Markdown(API_DOCS_MARKDOWN)

        with gr.TabItem("📤 Upload Transcript & Retrain Model"):
            gr.Markdown("### Stream Continuous Transcript Data & Trigger Automated Model Retraining")
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(label="Upload Student PDF Transcript", file_types=[".pdf"])
                    retrain_btn = gr.Button("Upload & Retrain Model", variant="secondary")
                with gr.Column():
                    retrain_output = gr.Textbox(label="Retraining Status", lines=4)

            retrain_btn.click(
                gradio_upload_and_retrain,
                inputs=[pdf_input],
                outputs=retrain_output,
                api_name="upload_and_retrain"
            )

if __name__ == "__main__":
    demo.launch()
