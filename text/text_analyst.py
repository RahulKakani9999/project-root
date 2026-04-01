"""
Text Analyst
Loads crime report CSV, cleans text, runs spaCy NER, HuggingFace sentiment
analysis and zero-shot classification, and outputs structured CSV.
"""

import os
import re
import csv
import glob
import uuid
import pandas as pd
import spacy
from transformers import pipeline


def load_models():
    """Load spaCy, sentiment, and zero-shot classification models."""
    nlp = spacy.load("en_core_web_sm")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )
    zeroshot_pipe = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
    )
    return nlp, sentiment_pipe, zeroshot_pipe


def clean_text(text):
    """Clean and normalize raw text."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s.,;:!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_location(nlp, text):
    """Extract location entities using spaCy NER."""
    doc = nlp(text[:2000])
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    return "; ".join(locations) if locations else "Unknown"


def extract_crime_type(nlp, text):
    """Extract crime-related entities and keywords from text."""
    doc = nlp(text[:2000])
    crime_keywords = {
        "Assault": ["assault", "attack", "battery", "hit", "punch", "stabbing"],
        "Robbery": ["robbery", "robbed", "holdup", "mugging", "stolen"],
        "Burglary": ["burglary", "break-in", "broke into", "trespassing"],
        "Theft": ["theft", "shoplifting", "larceny", "pickpocket", "stole"],
        "Fraud": ["fraud", "scam", "embezzlement", "forgery", "identity theft"],
        "Vandalism": ["vandalism", "graffiti", "property damage", "arson"],
        "Drug Offense": ["drug", "narcotics", "possession", "trafficking", "cocaine", "heroin"],
        "Homicide": ["murder", "homicide", "killing", "manslaughter"],
        "Domestic Violence": ["domestic", "abuse", "restraining order"],
        "Cybercrime": ["hacking", "phishing", "cyber", "ransomware", "data breach"],
    }
    text_lower = text.lower()
    for crime_type, keywords in crime_keywords.items():
        if any(kw in text_lower for kw in keywords):
            return crime_type
    return "Other"


def analyze_sentiment(sentiment_pipe, text):
    """Run sentiment analysis on text."""
    result = sentiment_pipe(text[:512])[0]
    return result["label"]


def classify_topic(zeroshot_pipe, text):
    """Run zero-shot classification to identify the topic."""
    candidate_labels = [
        "Public Safety",
        "Property Crime",
        "Violent Crime",
        "White Collar Crime",
        "Drug-Related Crime",
        "Traffic Incident",
        "Domestic Dispute",
        "Community Report",
    ]
    result = zeroshot_pipe(text[:512], candidate_labels)
    return result["labels"][0]


def assign_severity(crime_type, sentiment):
    """Assign a severity label based on crime type and sentiment."""
    high_severity = {"Homicide", "Assault", "Drug Offense", "Domestic Violence"}
    medium_severity = {"Robbery", "Burglary", "Fraud", "Cybercrime"}
    low_severity = {"Theft", "Vandalism"}

    if crime_type in high_severity:
        return "High"
    if crime_type in medium_severity:
        return "High" if sentiment == "NEGATIVE" else "Medium"
    if crime_type in low_severity:
        return "Medium" if sentiment == "NEGATIVE" else "Low"
    return "Medium"


def find_input_csv(text_dir):
    """Find a crime report CSV file in the directory."""
    csv_files = sorted(glob.glob(os.path.join(text_dir, "*.csv")))
    for f in csv_files:
        if "text_analysis" not in os.path.basename(f):
            return f
    return None


def load_crime_reports(csv_path):
    """Load crime reports from a CSV file."""
    df = pd.read_csv(csv_path)
    text_col = None
    for col in df.columns:
        if col.lower() in ("text", "report", "description", "narrative", "content", "details"):
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]
    return df[text_col].dropna().tolist()


def process_reports(text_dir):
    """Process crime report CSV and return analysis rows."""
    csv_path = find_input_csv(text_dir)
    if csv_path is None:
        return None

    reports = load_crime_reports(csv_path)
    if not reports:
        return None

    nlp, sentiment_pipe, zeroshot_pipe = load_models()
    rows = []

    for raw_text in reports:
        text = clean_text(raw_text)
        if not text:
            continue

        text_id = str(uuid.uuid4())[:8].upper()
        crime_type = extract_crime_type(nlp, text)
        location = extract_location(nlp, text)
        sentiment = analyze_sentiment(sentiment_pipe, text)
        topic = classify_topic(zeroshot_pipe, text)
        severity = assign_severity(crime_type, sentiment)

        rows.append({
            "Text_ID": text_id,
            "Crime_Type": crime_type,
            "Location_Entity": location,
            "Sentiment": sentiment,
            "Topic": topic,
            "Severity_Label": severity,
        })

    return rows if rows else None


def generate_demo_rows():
    """Generate 5 sample rows for demo mode."""
    return [
        {
            "Text_ID": "DEMO-001",
            "Crime_Type": "Robbery",
            "Location_Entity": "Downtown Los Angeles",
            "Sentiment": "NEGATIVE",
            "Topic": "Violent Crime",
            "Severity_Label": "High",
        },
        {
            "Text_ID": "DEMO-002",
            "Crime_Type": "Burglary",
            "Location_Entity": "Oak Park, Chicago",
            "Sentiment": "NEGATIVE",
            "Topic": "Property Crime",
            "Severity_Label": "Medium",
        },
        {
            "Text_ID": "DEMO-003",
            "Crime_Type": "Fraud",
            "Location_Entity": "Manhattan, New York",
            "Sentiment": "NEGATIVE",
            "Topic": "White Collar Crime",
            "Severity_Label": "High",
        },
        {
            "Text_ID": "DEMO-004",
            "Crime_Type": "Vandalism",
            "Location_Entity": "Riverside, San Diego",
            "Sentiment": "NEGATIVE",
            "Topic": "Property Crime",
            "Severity_Label": "Medium",
        },
        {
            "Text_ID": "DEMO-005",
            "Crime_Type": "Drug Offense",
            "Location_Entity": "Midtown Atlanta",
            "Sentiment": "NEGATIVE",
            "Topic": "Drug-Related Crime",
            "Severity_Label": "High",
        },
    ]


def write_csv(rows, output_path):
    """Write analysis rows to a CSV file."""
    fieldnames = ["Text_ID", "Crime_Type", "Location_Entity", "Sentiment", "Topic", "Severity_Label"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output written to {output_path} ({len(rows)} rows)")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir = os.path.join(project_root, "sample_data", "text")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "text_analysis.csv")

    rows = process_reports(input_dir)

    if rows is None:
        print("No crime report CSV found. Running in demo mode with sample data.")
        rows = generate_demo_rows()

    write_csv(rows, output_path)


if __name__ == "__main__":
    main()
