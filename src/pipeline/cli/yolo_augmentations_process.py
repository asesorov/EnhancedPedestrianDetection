import sys
import argparse
import onnxruntime as ort
import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1].joinpath('augmentation')))
from image_albumenation import transform


def preprocess_frame(frame):
    # Convert BGR to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize the frame to 640x640
    frame_resized = cv2.resize(frame_rgb, (640, 640))

    # Normalize pixel values to the range [0, 1]
    frame_normalized = frame_resized / 255.0

    return frame_normalized

def process_video(video_path, output_path, yolo_model_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    yolo_session = YOLO(yolo_model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO for detection
        results = yolo_session.predict(frame)                    # run prediction on img
        for result in results:                                         # iterate results
            boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy
            if boxes:
                for box in boxes:                                          # iterate boxes
                    r = box.xyxy[0].astype(int)                            # get corner points as int
                    cv2.rectangle(frame, r[:2], r[2:], (255, 255, 255), 2)

        # Write the processed frame to the output video
        out.write(frame)

    # Release video capture and writer
    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Video with YOLO and Lowlight Model')
    parser.add_argument('video_path', type=str, help='Path to the input video')
    parser.add_argument('output_path', type=str, help='Path to the output video')
    parser.add_argument('yolo_model_path', type=str, help='Path to the YOLO model')

    args = parser.parse_args()

    process_video(args.video_path, args.output_path, args.yolo_model_path,)
