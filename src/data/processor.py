import os
import glob
import shutil
import pypdfium2 as pdfium
from src.config import (
    DATA_DIR, OUTPUT_JSON_PATH, MAX_SEMESTER_HOURS,
    COURSE_CODE_REGEX, HOURS_REGEX, YEAR_REGEX,
    ARABIC_SEMESTER_KEYWORDS, REVERSED_ARABIC_KEYWORDS,
    ENGLISH_SEMESTER_KEYWORDS, FAIL_KEYWORDS
)
from src.utils import load_json, save_json

def is_header_line(line_clean):
    lower_line = line_clean.lower()
    if any(kw in line_clean for kw in ARABIC_SEMESTER_KEYWORDS):
        return True
    if any(kw in line_clean for kw in REVERSED_ARABIC_KEYWORDS):
        return True
    if any(kw in lower_line for kw in ENGLISH_SEMESTER_KEYWORDS):
        return True
    if YEAR_REGEX.search(line_clean):
        return True
    return False

def parse_pdf_transcript(pdf_path):
    semesters = []
    current_semester = {"name": "Semester 1", "courses": [], "passed": [], "failed": []}
    semester_idx = 1
    accumulated_hours = 0

    pdf = pdfium.PdfDocument(pdf_path)
    try:
        for page_idx in range(len(pdf)):
            page = pdf.get_page(page_idx)
            textpage = page.get_textpage()
            page_text = textpage.get_text_bounded()
            textpage.close()
            page.close()

            lines = page_text.split('\n')
            for line in lines:
                line_clean = line.strip()
                if not line_clean:
                    continue

                if is_header_line(line_clean) and current_semester["courses"]:
                    semesters.append(current_semester)
                    semester_idx += 1
                    current_semester = {"name": f"Semester {semester_idx}", "courses": [], "passed": [], "failed": []}
                    accumulated_hours = 0

                matches = COURSE_CODE_REGEX.findall(line_clean)
                if matches:
                    for dept, num in matches:
                        code = f"{dept.upper()}{num}"
                        is_failed = any(kw in line_clean for kw in FAIL_KEYWORDS) or line_clean.startswith("X")
                        
                        hours_match = HOURS_REGEX.search(line_clean)
                        hours = int(hours_match.group(1)) if hours_match else 3

                        existing_codes = [c["code"] for c in current_semester["courses"]]
                        if code in existing_codes or accumulated_hours + hours > MAX_SEMESTER_HOURS:
                            if current_semester["courses"]:
                                semesters.append(current_semester)
                                semester_idx += 1
                                current_semester = {"name": f"Semester {semester_idx}", "courses": [], "passed": [], "failed": []}
                                accumulated_hours = 0
                                existing_codes = []

                        if code not in existing_codes:
                            current_semester["courses"].append({"code": code, "hours": hours, "passed": not is_failed})
                            accumulated_hours += hours
                            if not is_failed:
                                current_semester["passed"].append(code)
                            else:
                                current_semester["failed"].append(code)
    finally:
        pdf.close()

    if current_semester["courses"]:
        semesters.append(current_semester)

    return {
        "pdf_filename": os.path.basename(pdf_path),
        "total_semesters": len(semesters),
        "semesters": semesters
    }

def process_all_transcripts(data_dir=DATA_DIR, output_json=OUTPUT_JSON_PATH):
    os.makedirs(data_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        root_pdfs = glob.glob("*.pdf")
        for pdf in root_pdfs:
            dest = os.path.join(data_dir, os.path.basename(pdf))
            if not os.path.exists(dest):
                shutil.copy(pdf, dest)
        pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))

    all_students_data = []
    training_samples = []
    extracted_catalog = {}

    for pdf_path in pdf_files:
        parsed_student = parse_pdf_transcript(pdf_path)
        all_students_data.append(parsed_student)
        accumulated_passed = []

        for sem_idx, semester in enumerate(parsed_student["semesters"]):
            for c in semester["courses"]:
                code = c["code"]
                if code not in extracted_catalog:
                    extracted_catalog[code] = {"hours": c["hours"]}

            current_history = list(accumulated_passed)

            for pos_code in semester["passed"]:
                hist = current_history if current_history else [c for c in semester["passed"] if c != pos_code]
                training_samples.append({
                    "student_history": list(hist) if hist else ["<PAD>"],
                    "candidate_id": pos_code,
                    "target_score": 1.0,
                    "source_pdf": parsed_student["pdf_filename"]
                })

            for neg_code in semester["failed"]:
                hist = current_history if current_history else [c for c in semester["passed"]]
                training_samples.append({
                    "student_history": list(hist) if hist else ["<PAD>"],
                    "candidate_id": neg_code,
                    "target_score": 0.0,
                    "source_pdf": parsed_student["pdf_filename"]
                })

            accumulated_passed.extend(semester["passed"])

        for sem_idx, semester in enumerate(parsed_student["semesters"]):
            passed_so_far = [c for s in parsed_student["semesters"][:sem_idx + 1] for c in s["passed"]]
            if sem_idx < len(parsed_student["semesters"]) - 1:
                future_courses = [c["code"] for s in parsed_student["semesters"][sem_idx + 1:] for c in s["courses"]]
                for fc in future_courses[:3]:
                    training_samples.append({
                        "student_history": list(passed_so_far),
                        "candidate_id": fc,
                        "target_score": 0.2,
                        "source_pdf": parsed_student["pdf_filename"]
                    })

    output_dataset = {
        "extracted_students_count": len(all_students_data),
        "total_training_samples": len(training_samples),
        "course_catalog": extracted_catalog,
        "students": all_students_data,
        "training_samples": training_samples
    }

    save_json(output_dataset, output_json)
    return output_dataset

def load_training_dataset(output_json=OUTPUT_JSON_PATH, data_dir=DATA_DIR, force_reprocess=False):
    actual_pdf_files = glob.glob(os.path.join(data_dir, "*.pdf")) or glob.glob("*.pdf")
    if force_reprocess or not os.path.exists(output_json):
        return process_all_transcripts(data_dir=data_dir, output_json=output_json)
        
    dataset = load_json(output_json)
    if not dataset or not dataset.get("training_samples") or len(actual_pdf_files) != dataset.get("extracted_students_count", 0):
        dataset = process_all_transcripts(data_dir=data_dir, output_json=output_json)
        
    return dataset
