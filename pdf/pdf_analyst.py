"""
PDF Analyst
Extracts text from PDFs using pdfplumber with OCR fallback (pytesseract + pdf2image),
runs spaCy NER to identify entities, and outputs structured CSV.
"""

import os
import re
import csv
import glob
import uuid
import pdfplumber
import pytesseract
import spacy
from pdf2image import convert_from_path


def load_spacy_model():
    """Load spaCy NER model."""
    return spacy.load("en_core_web_sm")


def extract_text_pdfplumber(filepath):
    """Extract text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_ocr(filepath):
    """Fallback: extract text from a PDF using OCR via pytesseract + pdf2image."""
    images = convert_from_path(filepath)
    text_parts = []
    for image in images:
        text_parts.append(pytesseract.image_to_string(image))
    return "\n".join(text_parts)


def extract_text(filepath):
    """Extract text from a PDF, falling back to OCR if pdfplumber yields no content."""
    text = extract_text_pdfplumber(filepath)
    if not text.strip():
        text = extract_text_ocr(filepath)
    return text


def extract_date(nlp, text):
    """Extract the first date entity from text using spaCy."""
    doc = nlp(text[:2000])
    for ent in doc.ents:
        if ent.label_ == "DATE":
            return ent.text
    date_pattern = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)
    if date_pattern:
        return date_pattern.group()
    return "Unknown"


def extract_department(nlp, text):
    """Extract department/organization from text using spaCy."""
    doc = nlp(text[:2000])
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return orgs[0] if orgs else "Unknown"


def extract_program(nlp, text):
    """Extract program or project references from text."""
    doc = nlp(text[:3000])
    programs = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "PRODUCT", "LAW", "EVENT")]
    seen = set()
    unique = []
    for p in programs:
        if p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)
    return "; ".join(unique[:3]) if unique else "General"


def classify_doc_type(text):
    """Classify the document type based on keyword matching."""
    text_lower = text[:3000].lower()
    doc_types = {
        "Financial Report": ["budget", "expenditure", "revenue", "fiscal", "financial"],
        "Policy Document": ["policy", "regulation", "compliance", "directive", "guideline"],
        "Incident Report": ["incident", "accident", "injury", "emergency", "hazard"],
        "Meeting Minutes": ["minutes", "meeting", "agenda", "resolution", "attendees"],
        "Research Report": ["research", "study", "findings", "analysis", "methodology"],
    }
    for doc_type, keywords in doc_types.items():
        if any(kw in text_lower for kw in keywords):
            return doc_type
    return "General Document"


def extract_key_detail(text):
    """Extract a key detail summary from the first meaningful sentences."""
    sentences = re.split(r'[.!?]+', text.strip())
    meaningful = [s.strip() for s in sentences if len(s.strip()) > 30]
    if meaningful:
        detail = meaningful[0]
        return detail[:200] + "..." if len(detail) > 200 else detail
    return "No detail available"


def process_pdf_files(pdf_dir):
    """Process all PDF files in the directory and return analysis rows."""
    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

    if not pdf_files:
        return None

    nlp = load_spacy_model()
    rows = []

    for filepath in pdf_files:
        report_id = str(uuid.uuid4())[:8].upper()
        text = extract_text(filepath)

        if not text.strip():
            continue

        department = extract_department(nlp, text)
        doc_type = classify_doc_type(text)
        date = extract_date(nlp, text)
        program = extract_program(nlp, text)
        key_detail = extract_key_detail(text)

        rows.append({
            "Report_ID": report_id,
            "Department": department,
            "Doc_Type": doc_type,
            "Date": date,
            "Program": program,
            "Key_Detail": key_detail,
        })

    return rows


def generate_demo_rows():
    """Generate 5 sample rows for demo mode."""
    return [
        {
            "Report_ID": "DEMO-001",
            "Department": "Department of Health and Human Services",
            "Doc_Type": "Policy Document",
            "Date": "January 15, 2025",
            "Program": "Medicare Expansion Initiative",
            "Key_Detail": "New guidelines for expanding Medicare coverage to include telehealth services in rural communities across 12 states",
        },
        {
            "Report_ID": "DEMO-002",
            "Department": "Federal Emergency Management Agency",
            "Doc_Type": "Incident Report",
            "Date": "March 3, 2025",
            "Program": "National Disaster Relief Fund",
            "Key_Detail": "Assessment of flood damage in the Mississippi River basin affecting approximately 15,000 households",
        },
        {
            "Report_ID": "DEMO-003",
            "Department": "Department of Education",
            "Doc_Type": "Financial Report",
            "Date": "February 28, 2025",
            "Program": "STEM Education Grant Program",
            "Key_Detail": "Fiscal year budget allocation of $2.4 billion for STEM education grants distributed across 500 school districts",
        },
        {
            "Report_ID": "DEMO-004",
            "Department": "Environmental Protection Agency",
            "Doc_Type": "Research Report",
            "Date": "December 10, 2024",
            "Program": "Clean Water Act Enforcement",
            "Key_Detail": "Study findings on water contamination levels in industrial zones showing 23% reduction after new regulations",
        },
        {
            "Report_ID": "DEMO-005",
            "Department": "Department of Transportation",
            "Doc_Type": "Meeting Minutes",
            "Date": "January 22, 2025",
            "Program": "National Infrastructure Plan",
            "Key_Detail": "Committee approved the allocation of funds for bridge repair projects in the northeastern corridor",
        },
    ]


def write_csv(rows, output_path):
    """Write analysis rows to a CSV file."""
    fieldnames = ["Report_ID", "Department", "Doc_Type", "Date", "Program", "Key_Detail"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output written to {output_path} ({len(rows)} rows)")


def main():
    pdf_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(pdf_dir, "pdf_analysis.csv")

    rows = process_pdf_files(pdf_dir)

    if rows is None:
        print("No PDF files found. Running in demo mode with sample data.")
        rows = generate_demo_rows()

    write_csv(rows, output_path)


if __name__ == "__main__":
    main()
