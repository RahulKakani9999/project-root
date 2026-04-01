"""
Integration Merge
Loads all 5 analyst output CSVs, merges them using incident_mapping.csv,
fills missing values with N/A, computes Final_Severity, and saves
final_incident_dataset.csv.
"""

import os
import csv
import pandas as pd


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
        print(f"  Loaded {label}: {len(df)} rows")
        return df
    print(f"  Warning: {label} not found at {path}, skipping.")
    return pd.DataFrame()


def generate_mapping(audio_df, pdf_df, image_df, video_df, text_df):
    """Generate incident_mapping.csv from the first 5 rows of each CSV."""
    rows = []
    for i in range(5):
        call_id = audio_df.iloc[i]["Call_ID"] if i < len(audio_df) else "N/A"
        report_id = pdf_df.iloc[i]["Report_ID"] if i < len(pdf_df) else "N/A"
        image_id = image_df.iloc[i]["Image_ID"] if i < len(image_df) else "N/A"
        # Video may have multiple rows per clip; get unique Clip_IDs
        unique_clips = video_df["Clip_ID"].unique() if not video_df.empty else []
        clip_id = unique_clips[i] if i < len(unique_clips) else (unique_clips[-1] if len(unique_clips) > 0 else "N/A")
        text_id = text_df.iloc[i]["Text_ID"] if i < len(text_df) else "N/A"
        rows.append({
            "Incident_ID": f"INC_{i+1:03d}",
            "Call_ID": call_id,
            "Report_ID": report_id,
            "Image_ID": image_id,
            "Clip_ID": clip_id,
            "Text_ID": text_id,
        })

    mapping_df = pd.DataFrame(rows)
    mapping_df.to_csv(MAPPING_PATH, index=False)
    print(f"  Generated mapping: {len(mapping_df)} incidents -> {MAPPING_PATH}")
    return mapping_df


def compute_final_severity(row):
    """Compute Final_Severity: High if urgency>0.8 or multiple negative signals,
    Medium if urgency>0.5, else Low."""
    negative_count = 0

    # Check audio urgency
    urgency = row.get("Urgency_Score")
    if pd.notna(urgency) and urgency != "N/A":
        urgency = float(urgency)
        if urgency > 0.8:
            return "High"
    else:
        urgency = 0.0

    # Count negative signals across sources
    # Audio sentiment
    audio_sentiment = row.get("Sentiment_audio")
    if pd.notna(audio_sentiment) and str(audio_sentiment).upper() == "NEGATIVE":
        negative_count += 1

    # Text sentiment
    text_sentiment = row.get("Sentiment_text")
    if pd.notna(text_sentiment) and str(text_sentiment).upper() == "NEGATIVE":
        negative_count += 1

    # Text severity label
    text_severity = row.get("Severity_Label")
    if pd.notna(text_severity) and str(text_severity) == "High":
        negative_count += 1

    # High-risk crime types
    crime_type = row.get("Crime_Type")
    high_crimes = {"Homicide", "Assault", "Robbery", "Drug Offense", "Domestic Violence"}
    if pd.notna(crime_type) and str(crime_type) in high_crimes:
        negative_count += 1

    # Video events
    event = row.get("Event_Detected")
    if pd.notna(event) and str(event) in ("Crowd Activity", "Vehicle Movement"):
        negative_count += 1

    # Image confidence (high confidence detection = more certain about scene)
    img_conf = row.get("Confidence_image")
    if pd.notna(img_conf) and isinstance(img_conf, str) and "avg=" in img_conf:
        try:
            avg_val = float(img_conf.split("avg=")[1].split(",")[0])
            if avg_val > 0.85:
                negative_count += 1
        except (ValueError, IndexError):
            pass

    # Multiple negative signals -> High
    if negative_count >= 2:
        return "High"

    # Moderate urgency -> Medium
    if urgency > 0.5:
        return "Medium"

    # Single negative signal -> Medium
    if negative_count == 1:
        return "Medium"

    return "Low"


def main():
    print("=" * 60)
    print("Integration Merge - Building Final Incident Dataset")
    print("=" * 60)

    # Load all 5 source CSVs
    print("\nLoading analyst outputs...")
    audio_df = load_csv(CSV_PATHS["audio"], "audio")
    pdf_df = load_csv(CSV_PATHS["pdf"], "pdf")
    image_df = load_csv(CSV_PATHS["image"], "image")
    video_df = load_csv(CSV_PATHS["video"], "video")
    text_df = load_csv(CSV_PATHS["text"], "text")

    # Generate incident mapping from first 5 rows of each CSV
    print("\nGenerating incident mapping...")
    mapping = generate_mapping(audio_df, pdf_df, image_df, video_df, text_df)

    # Aggregate video per Clip_ID (first event, max persons, mean confidence)
    if not video_df.empty:
        video_agg = video_df.groupby("Clip_ID").agg(
            Timestamp=("Timestamp", "first"),
            Event_Detected=("Event_Detected", "first"),
            Persons_Count=("Persons_Count", "max"),
            Video_Confidence=("Confidence", "mean"),
            Frame_Count=("Frame_ID", "count"),
        ).reset_index()
        video_agg["Video_Confidence"] = video_agg["Video_Confidence"].round(2)
    else:
        video_agg = pd.DataFrame()

    # Merge all sources onto the mapping
    print("\nMerging data sources...")
    merged = mapping.copy()

    # Audio merge
    if not audio_df.empty:
        merged = merged.merge(audio_df, on="Call_ID", how="left", suffixes=("", "_audio"))

    # PDF merge
    if not pdf_df.empty:
        merged = merged.merge(pdf_df, on="Report_ID", how="left", suffixes=("", "_pdf"))

    # Image merge
    if not image_df.empty:
        merged = merged.merge(image_df, on="Image_ID", how="left", suffixes=("", "_image"))

    # Video merge
    if not video_agg.empty:
        merged = merged.merge(video_agg, on="Clip_ID", how="left", suffixes=("", "_video"))

    # Text merge
    if not text_df.empty:
        merged = merged.merge(text_df, on="Text_ID", how="left", suffixes=("", "_text"))

    # Disambiguate Sentiment and Confidence columns with suffixes
    # Rename any bare Sentiment/Confidence that came from the first merge (audio)
    rename_map = {}
    cols = list(merged.columns)
    if "Sentiment" in cols and "Sentiment_text" in cols:
        rename_map["Sentiment"] = "Sentiment_audio"
    if "Confidence" in cols and "Confidence_image" not in cols:
        # If Confidence exists without _image suffix, check context
        pass
    # Rename Confidence columns for clarity
    if "Confidence" in cols:
        # This is from image (first merge that has Confidence)
        rename_map["Confidence"] = "Confidence_image"
    if rename_map:
        merged = merged.rename(columns=rename_map)

    # Fill missing values with N/A
    merged = merged.fillna("N/A")

    # Compute Final_Severity
    print("Computing Final_Severity...")
    merged["Final_Severity"] = merged.apply(compute_final_severity, axis=1)

    # Save output
    merged.to_csv(OUTPUT_PATH, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Final dataset saved to: {OUTPUT_PATH}")
    print(f"Total incidents: {len(merged)}")
    print(f"Total columns: {len(merged.columns)}")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nSeverity distribution:")
    for severity, count in merged["Final_Severity"].value_counts().items():
        print(f"  {severity}: {count}")
    print(f"\nPreview:")
    preview_cols = ["Incident_ID", "Call_ID", "Report_ID", "Image_ID", "Clip_ID", "Text_ID", "Final_Severity"]
    available = [c for c in preview_cols if c in merged.columns]
    print(merged[available].to_string(index=False))


if __name__ == "__main__":
    main()
