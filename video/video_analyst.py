"""
Video Analyst
Extracts frames from video using OpenCV every 15th frame, runs YOLOv8
object detection, performs motion detection, and outputs structured CSV.
"""

import os
import csv
import glob
import uuid
import cv2
import numpy as np
from ultralytics import YOLO


def load_model():
    """Load YOLOv8 model."""
    return YOLO("yolov8n.pt")


def detect_objects(model, frame):
    """Run YOLOv8 detection on a single frame."""
    results = model(frame, verbose=False)
    detections = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            label = result.names[cls_id]
            conf = round(float(boxes.conf[i]), 2)
            detections.append({"label": label, "confidence": conf})
    return detections


def detect_motion(prev_gray, curr_gray, threshold=25, min_area=500):
    """Detect motion between two grayscale frames."""
    if prev_gray is None:
        return False, 0.0
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > min_area)
    total_area = curr_gray.shape[0] * curr_gray.shape[1]
    motion_ratio = motion_area / total_area
    has_motion = motion_ratio > 0.01
    return has_motion, round(motion_ratio, 4)


def classify_event(detections, has_motion):
    """Classify the event based on detected objects and motion."""
    labels = {d["label"] for d in detections}
    person_count = sum(1 for d in detections if d["label"] == "person")

    if not has_motion and not detections:
        return "Static/Empty"
    if {"car", "truck", "bus"} & labels and has_motion:
        return "Vehicle Movement"
    if person_count >= 5:
        return "Crowd Activity"
    if person_count >= 1 and has_motion:
        return "Person Movement"
    if {"fire hydrant", "stop sign", "traffic light"} & labels:
        return "Street Infrastructure"
    if has_motion:
        return "General Motion"
    if detections:
        return "Static Objects"
    return "No Event"


def format_timestamp(frame_number, fps):
    """Convert frame number to HH:MM:SS.ms timestamp."""
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def process_video(model, filepath):
    """Process a single video file and return analysis rows."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    clip_id = str(uuid.uuid4())[:8].upper()
    rows = []
    prev_gray = None
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % 15 == 0:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

            detections = detect_objects(model, frame)
            has_motion, motion_ratio = detect_motion(prev_gray, curr_gray)
            event = classify_event(detections, has_motion)
            person_count = sum(1 for d in detections if d["label"] == "person")

            confidences = [d["confidence"] for d in detections]
            avg_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

            frame_id = f"F{frame_num:06d}"
            timestamp = format_timestamp(frame_num, fps)

            rows.append({
                "Clip_ID": clip_id,
                "Timestamp": timestamp,
                "Frame_ID": frame_id,
                "Event_Detected": event,
                "Persons_Count": person_count,
                "Confidence": avg_conf,
            })

            prev_gray = curr_gray

        frame_num += 1

    cap.release()
    return rows


def process_video_files(video_dir):
    """Process all video files in the directory and return analysis rows."""
    extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv")
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    video_files.sort()

    if not video_files:
        return None

    model = load_model()
    all_rows = []

    for filepath in video_files:
        rows = process_video(model, filepath)
        all_rows.extend(rows)

    return all_rows if all_rows else None


def generate_demo_rows():
    """Generate 5 sample rows for demo mode."""
    return [
        {
            "Clip_ID": "DEMO-VID1",
            "Timestamp": "00:00:00.00",
            "Frame_ID": "F000000",
            "Event_Detected": "Vehicle Movement",
            "Persons_Count": 2,
            "Confidence": 0.89,
        },
        {
            "Clip_ID": "DEMO-VID1",
            "Timestamp": "00:00:05.50",
            "Frame_ID": "F000165",
            "Event_Detected": "Person Movement",
            "Persons_Count": 3,
            "Confidence": 0.84,
        },
        {
            "Clip_ID": "DEMO-VID1",
            "Timestamp": "00:00:12.00",
            "Frame_ID": "F000360",
            "Event_Detected": "Crowd Activity",
            "Persons_Count": 7,
            "Confidence": 0.76,
        },
        {
            "Clip_ID": "DEMO-VID2",
            "Timestamp": "00:00:00.00",
            "Frame_ID": "F000000",
            "Event_Detected": "Static Objects",
            "Persons_Count": 0,
            "Confidence": 0.91,
        },
        {
            "Clip_ID": "DEMO-VID2",
            "Timestamp": "00:00:08.00",
            "Frame_ID": "F000240",
            "Event_Detected": "General Motion",
            "Persons_Count": 1,
            "Confidence": 0.68,
        },
    ]


def write_csv(rows, output_path):
    """Write analysis rows to a CSV file."""
    fieldnames = ["Clip_ID", "Timestamp", "Frame_ID", "Event_Detected", "Persons_Count", "Confidence"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Output written to {output_path} ({len(rows)} rows)")


def main():
    video_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(video_dir, "video_analysis.csv")

    rows = process_video_files(video_dir)

    if rows is None:
        print("No video files found. Running in demo mode with sample data.")
        rows = generate_demo_rows()

    write_csv(rows, output_path)


if __name__ == "__main__":
    main()
