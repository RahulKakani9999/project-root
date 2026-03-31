"""
Integration Merge
Loads all 5 analyst output CSVs, merges them using incident_mapping.csv,
fills missing values with N/A, computes Final_Severity, and saves
final_incident_dataset.csv.
"""

import os
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


def load_mapping():
    """Load the incident mapping CSV."""
    if not os.path.exists(MAPPING_PATH):
        raise FileNotFoundError(f"Incident mapping not found: {MAPPING_PATH}")
    df = pd.read_csv(MAPPING_PATH)
    print(f"  Loaded mapping: {len(df)} incidents")
    return df


def merge_source(mapping, source_df, mapping_key, source_key):
    """Merge a source DataFrame onto the mapping via key columns."""
    if source_df.empty:
        return mapping
    return mapping.merge(source_df, left_on=mapping_key, right_on=source_key, how="left")


def compute_final_severity(row):
    """Compute Final_Severity: High if urgency>0.8 or multiple negative signals,
    Medium if urgency>0.5, else Low."""
    negative_count = 0

    # Check audio urgency
    urgency = row.get("Urgency_Score")
    if pd.notna(urgency):
        urgency = float(urgency)
        if urgency > 0.8:
            return "High"
        if urgency > 0.5:
            # Still check for multiple negatives to potentially upgrade
            pass
    else:
        urgency = 0.0

    # Count negative signals across sources
    audio_sentiment = row.get("Sentiment")
    if pd.notna(audio_sentiment) and str(audio_sentiment).upper() == "NEGATIVE":
        negative_count += 1

    text_sentiment = row.get("Sentiment_text")
    if pd.notna(text_sentiment) and str(text_sentiment).upper() == "NEGATIVE":
        negative_count += 1

    text_severity = row.get("Severity_Label")
    if pd.notna(text_severity) and str(text_severity) == "High":
        negative_count += 1

    crime_type = row.get("Crime_Type")
    high_crimes = {"Homicide", "Assault", "Robbery", "Drug Offense", "Domestic Violence"}
    if pd.notna(crime_type) and str(crime_type) in high_crimes:
        negative_count += 1

    event = row.get("Event_Detected")
    if pd.notna(event) and str(event) in ("Crowd Activity", "Vehicle Movement"):
        negative_count += 1

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


def generate_demo_data():
    """Generate demo DataFrames for all 5 sources when no output CSVs exist."""
    audio_df = pd.DataFrame({
        "Call_ID": ["C001", "C002", "C003", "C004", "C005"],
        "Transcript": [
            "Fire at warehouse on 5th and Main. People trapped inside.",
            "Car accident on Highway 101 near exit 42. Multiple injuries.",
            "Flooding in downtown Springfield. Water levels rising fast.",
            "Medical emergency at Central Park. Person collapsed.",
            "Gas leak at apartment complex on Oak Avenue. Evacuating.",
        ],
        "Extracted_Event": ["Fire", "Car Accident", "Flooding", "Medical Emergency", "Gas Leak"],
        "Location": ["5th and Main Street", "Highway 101, Exit 42", "Downtown Springfield", "Central Park", "Oak Avenue"],
        "Sentiment": ["NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE"],
        "Urgency_Score": [0.95, 0.88, 0.91, 0.85, 0.82],
    })
    pdf_df = pd.DataFrame({
        "Report_ID": ["RPT_001", "RPT_002", "RPT_003", "RPT_004", "RPT_005"],
        "Department": [
            "Department of Health and Human Services",
            "Federal Emergency Management Agency",
            "Department of Education",
            "Environmental Protection Agency",
            "Department of Transportation",
        ],
        "Doc_Type": ["Policy Document", "Incident Report", "Financial Report", "Research Report", "Meeting Minutes"],
        "Date": ["January 15, 2025", "March 3, 2025", "February 28, 2025", "December 10, 2024", "January 22, 2025"],
        "Program": [
            "Medicare Expansion Initiative",
            "National Disaster Relief Fund",
            "STEM Education Grant Program",
            "Clean Water Act Enforcement",
            "National Infrastructure Plan",
        ],
        "Key_Detail": [
            "New guidelines for expanding Medicare coverage to telehealth services in rural communities",
            "Assessment of flood damage in Mississippi River basin affecting 15,000 households",
            "Fiscal year budget allocation of $2.4 billion for STEM education grants",
            "Water contamination levels in industrial zones showing 23% reduction",
            "Committee approved funds for bridge repair projects in northeastern corridor",
        ],
    })
    image_df = pd.DataFrame({
        "Image_ID": ["IMG_001", "IMG_002", "IMG_003", "IMG_004", "IMG_005"],
        "Scene_Type": ["Traffic Scene", "Indoor Scene", "Urban/Street Scene", "Dining Scene", "Office/Tech Scene"],
        "Objects_Detected": [
            "car (3); truck; traffic light (2); person (2)",
            "couch; tv; chair (2); potted plant",
            "person (5); bicycle (2); dog; bench",
            "cup (3); bowl (2); fork; knife; dining table",
            "laptop; keyboard; mouse; cell phone; person",
        ],
        "Bounding_Boxes": [
            "car:[120.5,200.3,340.1,380.7]; car:[400.0,210.5,580.2,390.0]",
            "couch:[30.0,250.0,450.5,500.0]; tv:[200.0,50.0,400.0,200.5]",
            "person:[100.0,80.0,180.5,350.0]; bicycle:[400.0,200.0,520.0,350.5]",
            "cup:[150.0,100.0,200.5,180.0]; bowl:[220.0,150.0,320.5,250.0]",
            "laptop:[100.0,150.0,400.5,380.0]; keyboard:[120.0,400.0,380.5,460.0]",
        ],
        "Confidence": [
            "avg=0.87, min=0.72, max=0.95",
            "avg=0.82, min=0.65, max=0.93",
            "avg=0.79, min=0.58, max=0.94",
            "avg=0.84, min=0.70, max=0.96",
            "avg=0.88, min=0.74, max=0.97",
        ],
    })
    video_df = pd.DataFrame({
        "Clip_ID": ["CAVIAR_01", "CAVIAR_01", "CAVIAR_02", "CAVIAR_02", "CAVIAR_03"],
        "Timestamp": ["00:00:00.00", "00:00:05.50", "00:00:00.00", "00:00:08.00", "00:00:00.00"],
        "Frame_ID": ["F000000", "F000165", "F000000", "F000240", "F000000"],
        "Event_Detected": ["Vehicle Movement", "Crowd Activity", "Person Movement", "General Motion", "Static Objects"],
        "Persons_Count": [2, 7, 3, 1, 0],
        "Confidence": [0.89, 0.76, 0.84, 0.68, 0.91],
    })
    text_df = pd.DataFrame({
        "Text_ID": ["TXT_001", "TXT_002", "TXT_003", "TXT_004", "TXT_005"],
        "Crime_Type": ["Robbery", "Burglary", "Fraud", "Vandalism", "Drug Offense"],
        "Location_Entity": [
            "Downtown Los Angeles",
            "Oak Park, Chicago",
            "Manhattan, New York",
            "Riverside, San Diego",
            "Midtown Atlanta",
        ],
        "Sentiment": ["NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE", "NEGATIVE"],
        "Topic": ["Violent Crime", "Property Crime", "White Collar Crime", "Property Crime", "Drug-Related Crime"],
        "Severity_Label": ["High", "Medium", "High", "Medium", "High"],
    })
    return audio_df, pdf_df, image_df, video_df, text_df


def main():
    print("=" * 60)
    print("Integration Merge - Building Final Incident Dataset")
    print("=" * 60)

    # Load mapping
    print("\nLoading incident mapping...")
    mapping = load_mapping()

    # Load source CSVs
    print("\nLoading analyst outputs...")
    sources = {name: load_csv(path, name) for name, path in CSV_PATHS.items()}

    # Check if any sources were found; if not, use demo mode
    all_empty = all(df.empty for df in sources.values())
    if all_empty:
        print("\nNo analyst output CSVs found. Running in demo mode with sample data.")
        audio_df, pdf_df, image_df, video_df, text_df = generate_demo_data()
    else:
        audio_df = sources["audio"]
        pdf_df = sources["pdf"]
        image_df = sources["image"]
        video_df = sources["video"]
        text_df = sources["text"]

    # Aggregate video per Clip_ID (take first event, max persons, mean confidence)
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
    merged = merge_source(merged, audio_df, "Call_ID", "Call_ID")
    merged = merge_source(merged, pdf_df, "Report_ID", "Report_ID")
    merged = merge_source(merged, image_df, "Image_ID", "Image_ID")
    merged = merge_source(merged, video_agg, "Clip_ID", "Clip_ID")

    # Text merge needs suffix handling since Sentiment column exists from audio
    if not text_df.empty:
        merged = merged.merge(text_df, left_on="Text_ID", right_on="Text_ID", how="left", suffixes=("", "_text"))
    else:
        merged = merged

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
    print(f"\nPreview (first 5 rows):")
    print(merged[["Incident_ID", "Call_ID", "Report_ID", "Image_ID", "Clip_ID", "Text_ID", "Final_Severity"]].to_string(index=False))


if __name__ == "__main__":
    main()
