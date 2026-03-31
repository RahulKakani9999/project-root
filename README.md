# Multimodal Crime/Incident Report Analyzer

A multimodal AI pipeline developed for the **AI for Engineers** course that analyzes crime and emergency incidents across five data modalities — audio, PDF documents, images, video, and text. The system processes each modality independently using state-of-the-art AI models, then merges all findings into a unified incident dataset with computed severity scores for downstream analysis and dashboarding.

## Project Description

Emergency response and law enforcement agencies generate data across multiple formats — 911 call recordings, scanned police reports, surveillance footage, crime scene photographs, and typed incident narratives. Analyzing these in isolation misses critical cross-modal patterns.

This project builds a **5-stage multimodal pipeline** that:

1. Ingests raw data from five different sources and formats
2. Applies AI models (Whisper, YOLOv8, spaCy, HuggingFace Transformers) to extract structured information
3. Produces per-modality CSV outputs with standardized schemas
4. Merges all outputs into a single incident-level dataset with a computed `Final_Severity` score
5. Enables dashboard visualization of incident patterns, severity distributions, and geographic hotspots

Each component runs independently and includes a **demo mode** with 5 sample rows when no input data is available.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: DATA INGESTION                      │
│  audio/*.wav  pdf/*.pdf  images/*.jpg  video/*.mp4  text/*.csv  │
└──────┬────────────┬──────────┬───────────┬──────────┬───────────┘
       │            │          │           │          │
       ▼            ▼          ▼           ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STAGE 2: AI PROCESSING                        │
│  Whisper     pdfplumber   YOLOv8     OpenCV+      spaCy NER    │
│  Speech-to-  + OCR        Object     YOLOv8       HuggingFace  │
│  Text        Fallback     Detection  Motion Det.  Zero-Shot    │
│  spaCy NER   spaCy NER    OCR        Event        Sentiment    │
│  Sentiment   Doc Classify Scene Type Classification            │
└──────┬────────────┬──────────┬───────────┬──────────┬───────────┘
       │            │          │           │          │
       ▼            ▼          ▼           ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 3: INFORMATION EXTRACTION                   │
│  emergency_   pdf_         image_      video_      text_        │
│  analysis.csv analysis.csv analysis.csv analysis.csv analysis.csv│
└──────┬────────────┬──────────┬───────────┬──────────┬───────────┘
       │            │          │           │          │
       └────────────┴──────────┴─────┬─────┴──────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│               STAGE 4: DATASET GENERATION                       │
│  integration/merge.py + incident_mapping.csv                    │
│  ► Joins all 5 CSVs on incident mapping keys                   │
│  ► Fills missing values with N/A                                │
│  ► Computes Final_Severity (High / Medium / Low)                │
│  ► Outputs final_incident_dataset.csv                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 5: DASHBOARD                              │
│  Visualization of severity distributions, incident timelines,   │
│  geographic hotspots, and cross-modal correlation analysis       │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (required for audio and video processing)
- Tesseract OCR (required for OCR fallback in PDF and image analysis)
- poppler-utils (required for pdf2image PDF-to-image conversion)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RahulKakani9999/project-root.git
cd project-root
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install all Python dependencies:
```bash
pip install -r requirements.txt
```

4. Download the spaCy English language model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

Each analyst script processes files in its own directory. If no input files are found, it automatically runs in **demo mode** with 5 sample rows.

### Step 1: Run Individual Analysts

```bash
# Audio: Transcribe 911 calls, extract entities, analyze sentiment
python audio/audio_analyst.py

# PDF: Extract text from police/government reports, classify documents
python pdf/pdf_analyst.py

# Images: Detect objects in crime scene photos, classify scenes
python images/image_analyst.py

# Video: Extract frames from surveillance footage, detect motion and events
python video/video_analyst.py

# Text: Analyze crime report narratives, classify crime types and severity
python text/text_analyst.py
```

### Step 2: Merge into Final Dataset

```bash
# Merge all 5 outputs into a unified incident dataset with Final_Severity
python integration/merge.py
```

The final output is saved to `integration/final_incident_dataset.csv`.

## Project Structure

```
project-root/
├── audio/
│   └── audio_analyst.py            # Whisper transcription + spaCy NER + sentiment
├── pdf/
│   └── pdf_analyst.py              # pdfplumber + OCR fallback + NER + doc classification
├── images/
│   └── image_analyst.py            # YOLOv8 object detection + scene classification + OCR
├── video/
│   └── video_analyst.py            # Frame extraction + YOLOv8 + motion detection
├── text/
│   └── text_analyst.py             # Crime report NER + sentiment + zero-shot classification
├── integration/
│   ├── merge.py                    # Merges all 5 CSVs into final incident dataset
│   └── incident_mapping.csv        # Maps incident IDs to per-modality component IDs
├── sample_data/
│   └── README.md                   # Dataset download links and instructions
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

## Technologies Used

| Category | Technology | Purpose |
|---|---|---|
| Speech-to-Text | OpenAI Whisper | Transcribe emergency audio recordings |
| PDF Processing | pdfplumber | Extract text from native PDF documents |
| OCR | pytesseract, pdf2image | Extract text from scanned PDFs and images |
| Object Detection | YOLOv8 (ultralytics) | Detect objects in images and video frames |
| Computer Vision | OpenCV | Frame extraction, motion detection, image I/O |
| NLP - NER | spaCy (en_core_web_sm) | Named entity recognition across all text modalities |
| NLP - Sentiment | HuggingFace Transformers (DistilBERT) | Sentiment analysis on transcripts and reports |
| NLP - Classification | HuggingFace Transformers (BART-MNLI) | Zero-shot topic classification for crime reports |
| Deep Learning | PyTorch | Backend for Whisper, YOLOv8, and Transformer models |
| NLP Toolkit | NLTK | Text preprocessing and tokenization |
| Data Processing | pandas, NumPy | Data manipulation, merging, and CSV I/O |
| Image Processing | Pillow, imageio | Image loading and format conversion |
| Video Processing | moviepy | Video editing and format support |

## License

This project is open source and available under the [MIT License](LICENSE).
