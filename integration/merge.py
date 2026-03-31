"""
Integration Merge
Loads all 5 analyst output CSVs, merges them using incident_mapping.csv,
computes Final_Severity, and saves final_incident_dataset.csv.
"""

import os
import pandas as pd


# Paths relative to this script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CSV_PATHS = {
    "audio": os.path.join(PROJECT_DIR, "audio", "emergency_analysis.csv"),
    "pdf": os.path.join(PROJECT_DIR, "pdf", "pdf_analysis.csv"),
    "image": os.path.join(PROJECT_DIR, "images", "image_analysis.csv"),
    "video": os.path.join(PROJECT_DIR, "video", "video_analysis.csv"),
    "text": os.path.join(PROJECT_DIR, "text", "text_analysis.csv"),
}

MAPPING_PATH = os.path.join(BASE_DIR, "incident_mapping.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "final_incident_dataset.csv")


def load_csv(path, label):
    """Load a CSV file, returning an empty DataFrame if not found."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded {label}: {len(df)} rows from {path}")
        return df
    print(f"Warning: {label} not found at {path}, skipping.")
    return pd.DataFrame()


def load_all_sources():
    """Load all 5 analyst output CSVs."""
    return {name: load_csv(path, name) for name, path in CSV_PATHS.items()}


def load_mapping():
    """Load the incident mapping CSV."""
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Incident mapping not found: {MAPPING_PATH}")
    df = pd.read_csv(MAPPING_PATH)
    print(f"Loaded mapping: {len(df)} incidents")
    return df


def merge_audio(mapping, audio_df):
    """Merge audio analysis data onto the mapping."""
    if audio_df.empty:
        return mapping
    audio_cols = audio_df.rename(columns={
        "Call_ID": "Audio_Call_ID",
        "Transcript": "Audio_Transcript",
        "Extracted_Event": "Audio_Event",
        "Location": "Audio_Location",
        "Sentiment": "Audio_Sentiment",
        "Urgency_Score": "Audio_Urgency",
    })
    return mapping.merge(audio_cols, on="Audio_Call_ID", how="left")


def merge_pdf(mapping, pdf_df):
    """Merge PDF analysis data onto the mapping."""
    if pdf_df.empty:
        return mapping
    pdf_cols = pdf_df.rename(columns={
        "Report_ID": "PDF_Report_ID",
        "Department": "PDF_Department",
        "Doc_Type": "PDF_Doc_Type",
        "Date": "PDF_Date",
        "Program": "PDF_Program",
        "Key_Detail": "PDF_Key_Detail",
    })
    return mapping.merge(pdf_cols, on="PDF_Report_ID", how="left")


def merge_image(mapping, image_df):
    """Merge image analysis data onto the mapping."""
    if image_df.empty:
        return mapping
    image_cols = image_df.rename(columns={
        "Image_ID": "Image_ID",
        "Scene_Type": "Image_Scene_Type",
        "Objects_Detected": "Image_Objects",
        "Bounding_Boxes": "Image_Bboxes",
        "Confidence": "Image_Confidence",
    })
    return mapping.merge(image_cols, on="Image_ID", how="left")


def merge_video(mapping, video_df):
    """Merge video analysis data onto the mapping (aggregate per clip)."""
    if video_df.empty:
        return mapping
    # Aggregate video rows per Clip_ID: take the most significant event
    agg = video_df.groupby("Clip_ID").agg(
        Video_Event=("Event_Detected", "first"),
        Video_Max_Persons=("Persons_Count", "max"),
        Video_Avg_Confidence=("Confidence", "mean"),
        Video_Frame_Count=("Frame_ID", "count"),
    ).reset_index()
    agg["Video_Avg_Confidence"] = agg["Video_Avg_Confidence"].round(2)
    agg = agg.rename(columns={"Clip_ID": "Video_Clip_ID"})
    return mapping.merge(agg, on="Video_Clip_ID", how="left")


def merge_text(mapping, text_df):
    """Merge text analysis data onto the mapping."""
    if text_df.empty:
        return mapping
    text_cols = text_df.rename(columns={
        "Text_ID": "Text_ID",
        "Crime_Type": "Text_Crime_Type",
        "Location_Entity": "Text_Location",
        "Sentiment": "Text_Sentiment",
        "Topic": "Text_Topic",
        "Severity_Label": "Text_Severity",
    })
    return mapping.merge(text_cols, on="Text_ID", how="left")


def compute_final_severity(row):
    """Compute Final_Severity from all available severity signals."""
    severity_score = 0.0
    weight_total = 0.0

    # Audio urgency (0-1 scale)
    audio_urgency = row.get("Audio_Urgency")
    if pd.notna(audio_urgency):
        severity_score += float(audio_urgency) * 3.0
        weight_total += 3.0

    # Audio sentiment
    audio_sent = row.get("Audio_Sentiment")
    if pd.notna(audio_sent) and audio_sent == "NEGATIVE":
        severity_score += 0.8 * 1.0
        weight_total += 1.0

    # Text severity label
    text_sev = row.get("Text_Severity")
    if pd.notna(text_sev):
        sev_map = {"High": 1.0, "Medium": 0.5, "Low": 0.2}
        severity_score += sev_map.get(text_sev, 0.5) * 2.0
        weight_total += 2.0

    # Text sentiment
    text_sent = row.get("Text_Sentiment")
    if pd.notna(text_sent) and text_sent == "NEGATIVE":
        severity_score += 0.7 * 1.0
        weight_total += 1.0

    # Video persons count
    max_persons = row.get("Video_Max_Persons")
    if pd.notna(max_persons):
        person_score = min(float(max_persons) / 10.0, 1.0)
        severity_score += person_score * 1.5
        weight_total += 1.5

    # Image confidence
    img_conf = row.get("Image_Confidence")
    if pd.notna(img_conf) and isinstance(img_conf, str) and "avg=" in img_conf:
        try:
            avg_val = float(img_conf.split("avg=")[1].split(",")[0])
            severity_score += avg_val * 1.0
            weight_total += 1.0
        except (ValueError, IndexError):
            pass

    if weight_total == 0:
        return "Medium"

    normalized = severity_score / weight_total
    if normalized >= 0.7:
        return "Critical"
    if normalized >= 0.5:
        return "High"
    if normalized >= 0.3:
        return "Medium"
    return "Low"


def main():
    print("=" * 60)
    print("Integration Merge - Building Final Incident Dataset")
    print("=" * 60)

    sources = load_all_sources()
    mapping = load_mapping()

    print("\nMerging data sources...")
    merged = mapping.copy()
    merged = merge_audio(merged, sources["audio"])
    merged = merge_pdf(merged, sources["pdf"])
    merged = merge_image(merged, sources["image"])
    merged = merge_video(merged, sources["video"])
    merged = merge_text(merged, sources["text"])

    print("Computing Final_Severity...")
    merged["Final_Severity"] = merged.apply(compute_final_severity, axis=1)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFinal dataset saved to {OUTPUT_PATH}")
    print(f"Total incidents: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nSeverity distribution:")
    print(merged["Final_Severity"].value_counts().to_string())


if __name__ == "__main__":
    main()
