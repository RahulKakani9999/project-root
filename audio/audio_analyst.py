"""
Emergency Audio Analyst
Transcribes emergency audio files using Whisper, extracts entities with spaCy,
runs sentiment analysis with HuggingFace, and outputs structured CSV.
"""

import os
import csv
import glob
import uuid
import whisper
import spacy
from transformers import pipeline


def load_models():
    """Load Whisper, spaCy, and HuggingFace sentiment models."""
    whisper_model = whisper.load_model("base")
    nlp = spacy.load("en_core_web_sm")
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return whisper_model, nlp, sentiment_pipeline


def transcribe_audio(whisper_model, filepath):
    """Transcribe an audio file using Whisper."""
    result = whisper_model.transcribe(filepath)
    return result["text"]


def extract_entities(nlp, text):
    """Extract event and location entities from transcript using spaCy."""
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]
    events = [ent.text for ent in doc.ents if ent.label_ in ("EVENT", "ORG", "NORP")]
    location = "; ".join(locations) if locations else "Unknown"
    event = "; ".join(events) if events else "General Emergency"
    return event, location


def analyze_sentiment(sentiment_pipeline, text):
    """Run sentiment analysis and derive an urgency score."""
    result = sentiment_pipeline(text[:512])[0]
    label = result["label"]
    score = result["score"]
    if label == "NEGATIVE":
        urgency = round(0.5 + score * 0.5, 2)
    else:
        urgency = round((1 - score) * 0.5, 2)
    return label, urgency


def process_audio_files(audio_dir):
    """Process all audio files in the directory and return analysis rows."""
    extensions = ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg")
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
    audio_files.sort()

    if not audio_files:
        return None

    whisper_model, nlp, sentiment_pipe = load_models()
    rows = []

    for filepath in audio_files:
        call_id = str(uuid.uuid4())[:8].upper()
        transcript = transcribe_audio(whisper_model, filepath)
        event, location = extract_entities(nlp, transcript)
        sentiment, urgency = analyze_sentiment(sentiment_pipe, transcript)

        rows.append({
            "Call_ID": call_id,
            "Transcript": transcript,
            "Extracted_Event": event,
            "Location": location,
            "Sentiment": sentiment,
            "Urgency_Score": urgency,
        })

    return rows


def generate_demo_rows():
    """Generate 5 sample rows for demo mode."""
    return [
        {
            "Call_ID": "DEMO-001",
            "Transcript": "There is a fire at the warehouse on 5th and Main Street. People are trapped inside.",
            "Extracted_Event": "Fire",
            "Location": "5th and Main Street",
            "Sentiment": "NEGATIVE",
            "Urgency_Score": 0.95,
        },
        {
            "Call_ID": "DEMO-002",
            "Transcript": "Car accident on Highway 101 near exit 42. Multiple vehicles involved, injuries reported.",
            "Extracted_Event": "Car Accident",
            "Location": "Highway 101, Exit 42",
            "Sentiment": "NEGATIVE",
            "Urgency_Score": 0.88,
        },
        {
            "Call_ID": "DEMO-003",
            "Transcript": "Flooding in the downtown area of Springfield. Water levels are rising rapidly.",
            "Extracted_Event": "Flooding",
            "Location": "Downtown Springfield",
            "Sentiment": "NEGATIVE",
            "Urgency_Score": 0.91,
        },
        {
            "Call_ID": "DEMO-004",
            "Transcript": "Medical emergency at Central Park. A person has collapsed and is unresponsive.",
            "Extracted_Event": "Medical Emergency",
            "Location": "Central Park",
            "Sentiment": "NEGATIVE",
            "Urgency_Score": 0.85,
        },
        {
            "Call_ID": "DEMO-005",
            "Transcript": "Gas leak reported at the apartment complex on Oak Avenue. Residents are evacuating.",
            "Extracted_Event": "Gas Leak",
            "Location": "Oak Avenue",
            "Sentiment": "NEGATIVE",
            "Urgency_Score": 0.82,
        },
    ]


def write_csv(rows, output_path):
    """Write analysis rows to a CSV file."""
    fieldnames = ["Call_ID", "Transcript", "Extracted_Event", "Location", "Sentiment", "Urgency_Score"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output written to {output_path} ({len(rows)} rows)")


def main():
    audio_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(audio_dir, "emergency_analysis.csv")

    rows = process_audio_files(audio_dir)

    if rows is None:
        print("No audio files found. Running in demo mode with sample data.")
        rows = generate_demo_rows()

    write_csv(rows, output_path)


if __name__ == "__main__":
    main()
