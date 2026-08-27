import os
import json
import gradio as gr
from src.api.routes import app as fastapi_app
from src.engine.recommender import recommend_registration_bundle
from src.data.processor import load_training_dataset
from src.config import OUTPUT_JSON_PATH

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

with gr.Blocks(title="Academic Advisor AI") as demo:
    gr.Markdown("# 🎓 Academic Advisor AI Recommendation Engine")
    gr.Markdown("Deep Learning model for university advising. **REST API for .NET backend is live at `/docs` (Swagger UI)**")

    with gr.Row():
        with gr.Column():
            passed_input = gr.Textbox(
                label="Passed Course Codes (comma-separated)",
                value="CS141, BS101, BS111, BS105, BS103, CS101, UNI008",
                placeholder="e.g. CS141, BS101, BS111"
            )
            hours_slider = gr.Slider(minimum=1, maximum=24, value=9, step=1, label="Max Credit Hours Limit")
            btn = gr.Button("Get Recommended Course Bundle", variant="primary")

        with gr.Column():
            output_json = gr.Code(label="Recommended Registration Bundle (JSON)", language="json")

    btn.click(gradio_recommend, inputs=[passed_input, hours_slider], outputs=output_json)

for route in fastapi_app.routes:
    if route not in demo.app.routes:
        demo.app.routes.append(route)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
