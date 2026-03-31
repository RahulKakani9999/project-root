# Multimodal Crime/Incident Report Analyzer

A multimodal AI pipeline that analyzes crime and emergency incidents across five data types — audio, PDF documents, images, video, and text — then merges all findings into a unified incident dataset with computed severity scores.

## Project Structure

```
project-root/
├── audio/
│   └── audio_analyst.py        # Whisper transcription + spaCy NER + sentiment
├── pdf/
│   └── pdf_analyst.py          # pdfplumber extraction + OCR fallback + NER
├── images/
│   └── image_analyst.py        # YOLOv8 object detection + scene classification + OCR
├── video/
│   └── video_analyst.py        # Frame extraction + YOLOv8 + motion detection
├── text/
│   └── text_analyst.py         # Crime report NER + sentiment + zero-shot classification
├── integration/
│   ├── merge.py                # Merges all 5 outputs into final dataset
│   └── incident_mapping.csv    # Maps incident IDs to component IDs
├── requirements.txt
└── README.md
```

## Components

### 1. Audio Analyst (`audio/audio_analyst.py`)
Transcribes emergency audio calls using OpenAI Whisper, extracts entities (events, locations) with spaCy, and runs HuggingFace sentiment analysis. Outputs `emergency_analysis.csv` with columns: `Call_ID`, `Transcript`, `Extracted_Event`, `Location`, `Sentiment`, `Urgency_Score`.

### 2. PDF Analyst (`pdf/pdf_analyst.py`)
Extracts text from PDF documents using pdfplumber with pytesseract + pdf2image OCR fallback for scanned files. Runs spaCy NER to identify departments, dates, and programs, and classifies document types. Outputs `pdf_analysis.csv` with columns: `Report_ID`, `Department`, `Doc_Type`, `Date`, `Program`, `Key_Detail`.

### 3. Image Analyst (`images/image_analyst.py`)
Runs YOLOv8 object detection via ultralytics, classifies scene types based on detected objects, and performs OCR with pytesseract. Outputs `image_analysis.csv` with columns: `Image_ID`, `Scene_Type`, `Objects_Detected`, `Bounding_Boxes`, `Confidence`.

### 4. Video Analyst (`video/video_analyst.py`)
Extracts every 15th frame from video using OpenCV, runs YOLOv8 detection, and performs motion detection via frame differencing. Classifies events such as Vehicle Movement, Crowd Activity, and Person Movement. Outputs `video_analysis.csv` with columns: `Clip_ID`, `Timestamp`, `Frame_ID`, `Event_Detected`, `Persons_Count`, `Confidence`.

### 5. Text Analyst (`text/text_analyst.py`)
Loads crime report CSVs, cleans text, extracts crime types and locations with spaCy NER, runs HuggingFace sentiment analysis and zero-shot classification (BART-MNLI). Outputs `text_analysis.csv` with columns: `Text_ID`, `Crime_Type`, `Location_Entity`, `Sentiment`, `Topic`, `Severity_Label`.

### Integration (`integration/merge.py`)
Loads all 5 analyst output CSVs, joins them via `incident_mapping.csv`, computes a weighted `Final_Severity` score (Critical/High/Medium/Low) from audio urgency, text severity, sentiment signals, video person counts, and image confidence. Saves `final_incident_dataset.csv`.

## Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (required for audio and video processing)
- Tesseract OCR (required for OCR fallback in PDF and image analysis)
- poppler-utils (required for pdf2image)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RahulKakani9999/project-root.git
cd project-root
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Running Each Script

Each analyst script runs independently. If no input files are found, it runs in **demo mode** with 5 sample rows.

```bash
# 1. Analyze audio files (or generate demo output)
python audio/audio_analyst.py

# 2. Analyze PDF documents (or generate demo output)
python pdf/pdf_analyst.py

# 3. Analyze images (or generate demo output)
python images/image_analyst.py

# 4. Analyze video files (or generate demo output)
python video/video_analyst.py

# 5. Analyze crime report text (or generate demo output)
python text/text_analyst.py

# 6. Merge all outputs into the final incident dataset
python integration/merge.py
```

## Output

The final merged dataset is saved to `integration/final_incident_dataset.csv` and includes fields from all five analysts plus a computed `Final_Severity` column (Critical, High, Medium, or Low).

## License

This project is open source and available under the [MIT License](LICENSE).
