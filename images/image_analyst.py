"""
Image Analyst
Runs YOLOv8 object detection using ultralytics, classifies scene type,
performs OCR with pytesseract, and outputs structured CSV.
"""

import os
import csv
import glob
import uuid
import cv2
import pytesseract
from ultralytics import YOLO
from PIL import Image


def load_model():
    """Load YOLOv8 model."""
    return YOLO("yolov8n.pt")


def detect_objects(model, filepath):
    """Run YOLOv8 object detection on an image."""
    results = model(filepath, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            label = result.names[cls_id]
            conf = round(float(boxes.conf[i]), 2)
            bbox = boxes.xyxy[i].tolist()
            bbox = [round(v, 1) for v in bbox]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": bbox,
            })
    return detections


def classify_scene(detections):
    """Classify scene type based on detected objects."""
    labels = {d["label"] for d in detections}

    scene_rules = [
        ({"car", "truck", "bus", "traffic light", "stop sign"}, "Traffic Scene"),
        ({"person", "bicycle", "dog", "cat", "bench"}, "Urban/Street Scene"),
        ({"airplane", "helicopter"}, "Aerial Scene"),
        ({"boat", "surfboard"}, "Maritime Scene"),
        ({"bed", "couch", "tv", "chair", "dining table"}, "Indoor Scene"),
        ({"sports ball", "tennis racket", "baseball bat", "skateboard"}, "Sports Scene"),
        ({"knife", "fork", "spoon", "bowl", "cup", "wine glass"}, "Dining Scene"),
        ({"laptop", "keyboard", "mouse", "cell phone", "monitor"}, "Office/Tech Scene"),
    ]

    for trigger_labels, scene_type in scene_rules:
        if labels & trigger_labels:
            return scene_type

    if detections:
        return "General Scene"
    return "Empty/Unclear Scene"


def run_ocr(filepath):
    """Run OCR on the image to extract any visible text."""
    image = Image.open(filepath)
    text = pytesseract.image_to_string(image).strip()
    return text if text else None


def format_objects(detections):
    """Format detected objects as a summary string."""
    if not detections:
        return "None"
    label_counts = {}
    for d in detections:
        label_counts[d["label"]] = label_counts.get(d["label"], 0) + 1
    parts = [f"{label} ({count})" if count > 1 else label for label, count in label_counts.items()]
    return "; ".join(parts)


def format_bboxes(detections):
    """Format bounding boxes as a string."""
    if not detections:
        return "None"
    parts = [f"{d['label']}:[{','.join(str(v) for v in d['bbox'])}]" for d in detections[:10]]
    return "; ".join(parts)


def format_confidence(detections):
    """Format confidence scores as a summary string."""
    if not detections:
        return "N/A"
    scores = [d["confidence"] for d in detections]
    avg = round(sum(scores) / len(scores), 2)
    return f"avg={avg}, min={min(scores)}, max={max(scores)}"


def process_image_files(image_dir):
    """Process all image files in the directory and return analysis rows."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
    image_files.sort()

    if not image_files:
        return None

    model = load_model()
    rows = []

    for filepath in image_files:
        image_id = str(uuid.uuid4())[:8].upper()
        detections = detect_objects(model, filepath)
        scene_type = classify_scene(detections)
        ocr_text = run_ocr(filepath)

        objects_str = format_objects(detections)
        if ocr_text:
            objects_str += f" | OCR: {ocr_text[:100]}"

        rows.append({
            "Image_ID": image_id,
            "Scene_Type": scene_type,
            "Objects_Detected": objects_str,
            "Bounding_Boxes": format_bboxes(detections),
            "Confidence": format_confidence(detections),
        })

    return rows


def generate_demo_rows():
    """Generate 5 sample rows for demo mode."""
    return [
        {
            "Image_ID": "DEMO-001",
            "Scene_Type": "Traffic Scene",
            "Objects_Detected": "car (3); truck; traffic light (2); person (2)",
            "Bounding_Boxes": "car:[120.5,200.3,340.1,380.7]; car:[400.0,210.5,580.2,390.0]; truck:[50.0,180.0,200.5,400.0]",
            "Confidence": "avg=0.87, min=0.72, max=0.95",
        },
        {
            "Image_ID": "DEMO-002",
            "Scene_Type": "Indoor Scene",
            "Objects_Detected": "couch; tv; chair (2); potted plant | OCR: Samsung 55\" UHD",
            "Bounding_Boxes": "couch:[30.0,250.0,450.5,500.0]; tv:[200.0,50.0,400.0,200.5]; chair:[500.0,220.0,620.5,480.0]",
            "Confidence": "avg=0.82, min=0.65, max=0.93",
        },
        {
            "Image_ID": "DEMO-003",
            "Scene_Type": "Urban/Street Scene",
            "Objects_Detected": "person (5); bicycle (2); dog; bench",
            "Bounding_Boxes": "person:[100.0,80.0,180.5,350.0]; person:[250.0,90.0,320.5,360.0]; bicycle:[400.0,200.0,520.0,350.5]",
            "Confidence": "avg=0.79, min=0.58, max=0.94",
        },
        {
            "Image_ID": "DEMO-004",
            "Scene_Type": "Dining Scene",
            "Objects_Detected": "cup (3); bowl (2); fork; knife; dining table",
            "Bounding_Boxes": "cup:[150.0,100.0,200.5,180.0]; cup:[300.0,110.0,350.5,190.0]; bowl:[220.0,150.0,320.5,250.0]",
            "Confidence": "avg=0.84, min=0.70, max=0.96",
        },
        {
            "Image_ID": "DEMO-005",
            "Scene_Type": "Office/Tech Scene",
            "Objects_Detected": "laptop; keyboard; mouse; cell phone; person | OCR: Q4 Revenue Report",
            "Bounding_Boxes": "laptop:[100.0,150.0,400.5,380.0]; keyboard:[120.0,400.0,380.5,460.0]; mouse:[420.0,410.0,470.5,450.0]",
            "Confidence": "avg=0.88, min=0.74, max=0.97",
        },
    ]


def write_csv(rows, output_path):
    """Write analysis rows to a CSV file."""
    fieldnames = ["Image_ID", "Scene_Type", "Objects_Detected", "Bounding_Boxes", "Confidence"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output written to {output_path} ({len(rows)} rows)")


def main():
    image_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(image_dir, "image_analysis.csv")

    rows = process_image_files(image_dir)

    if rows is None:
        print("No image files found. Running in demo mode with sample data.")
        rows = generate_demo_rows()

    write_csv(rows, output_path)


if __name__ == "__main__":
    main()
