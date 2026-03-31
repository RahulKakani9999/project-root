# Sample Data Downloads

This project requires data from five modalities. Below are the recommended sources for each dataset. Download the files and place them in the corresponding project directories.

## 1. Audio - Emergency 911 Calls

- **Source:** Kaggle - 911 Calls Wav2Vec2
- **URL:** https://kaggle.com/code/stpeteishii/911-calls-wav2vec2
- **Format:** WAV audio files
- **Place files in:** `audio/`
- **Description:** Emergency 911 call recordings used for speech-to-text transcription with Whisper, followed by entity extraction and sentiment analysis.

## 2. PDF - Government/Police Reports

- **Source:** MuckRock - Public Records and FOIA Documents
- **URL:** https://muckrock.com
- **Format:** PDF documents
- **Place files in:** `pdf/`
- **Description:** Government documents, police reports, and public records obtained through FOIA requests. The PDF analyst extracts text using pdfplumber with OCR fallback for scanned documents.

## 3. Images - Incident/Crime Scene Photos

- **Source:** Roboflow Universe - Fire/Incident Detection Datasets
- **URL:** https://universe.roboflow.com/search?q=fire
- **Format:** JPG, PNG images
- **Place files in:** `images/`
- **Description:** Annotated incident and fire detection image datasets. YOLOv8 performs object detection and scene classification on these images.

## 4. Video - Surveillance Footage

- **Source:** CAVIAR Dataset - EC Funded IST 2001 37540 Project
- **URL:** https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/
- **Format:** MP4, AVI video files
- **Place files in:** `video/`
- **Description:** Surveillance video sequences from the CAVIAR project recorded in a shopping centre and lobby. Used for frame extraction, YOLOv8 object detection, and motion analysis.

## 5. Text - Crime Report Narratives

- **Source:** Kaggle - Crime Report Dataset
- **URL:** https://kaggle.com/datasets/cameliasiadat/crimereport
- **Format:** CSV files
- **Place files in:** `text/`
- **Description:** Structured crime report narratives with incident descriptions. The text analyst performs NER, sentiment analysis, and zero-shot crime type classification.

## Notes

- All scripts include a **demo mode** that generates 5 sample output rows if no input files are found. You can run the full pipeline without downloading any data.
- After placing files in their respective directories, run each analyst script followed by `python integration/merge.py` to generate the final merged dataset.
- Ensure file formats match the supported extensions listed in each analyst script.
