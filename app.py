
import sys
sys.path.append("/content/CS420-Helmet-Detection-with-YOLO_The_app_part/backend/source")
sys.path.append("/content/CS420-Helmet-Detection-with-YOLO_The_app_part/backend/source/utils")
sys.path.append("/content/CS420-Helmet-Detection-with-YOLO_The_app_part/deep_sort_realtime")
sys.path.append("/content/CS420-Helmet-Detection-with-YOLO_The_app_part/deep_sort_realtime/deep_sort_realtime")

import os
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
import streamlit as st
from backend.source.test import run_on_frame, create_models
from backend.source.utils.visualize import plot_bbox
from deep_sort_realtime.deepsort_tracker import DeepSort

# Utility Functions
def clip_bbox(bbox, frame_width, frame_height):
  x1 = max(0, min(bbox[0], frame_width - 1))
  y1 = max(0, min(bbox[1], frame_height - 1))
  x2 = max(0, min(bbox[2], frame_width - 1))
  y2 = max(0, min(bbox[3], frame_height - 1))
  return [x1, y1, x2, y2]

def xyxy2ltwh(box):
  x1, y1, x2, y2 = box
  w, h = x2 - x1, y2 - y1
  return [x1, y1, w, h]

# Streamlit Configuration
st.set_page_config(page_title="Helmet Detection", layout="wide")

# Sidebar Settings
# st.sidebar.header("Tracker Settings")
# iou_thr = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5)
# skip_box_thr = st.sidebar.slider("Skip Box Threshold", 0.0, 1.0, 0.1)
# confidence_thr = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
# max_age = st.sidebar.number_input("Max Age for Tracker", min_value=1, max_value=100, value=30)
# n_init = st.sidebar.number_input("N Init for Tracker", min_value=1, max_value=10, value=3)
# max_cosine_distance = st.sidebar.slider("Max Cosine Distance", 0.0, 1.0, 0.5)

# Load Model Weights
def get_model_weights():
  weights_dir = "weights"
  return [
      os.path.join(weights_dir, model_name)
      for model_name in os.listdir(weights_dir)
      if model_name.endswith(".pt")
  ]

# Initialize Models
model_weights_list = get_model_weights()
models, model_names = create_models(model_weights_list)
# File Upload
uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
if uploaded_file:
    # Save uploaded file to a temporary location
    temp_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(uploaded_file.read())
    file_path = temp_file.name

    # Initialize Tracker
    tracker = DeepSort(
        max_age=20,
        n_init=3,
        max_cosine_distance=0.35,
        max_iou_distance=0.8,
        nms_max_overlap=0.7,
    )

    # Process Video
    cap = cv2.VideoCapture(file_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = min(30, frame_rate)
    frame_interval = frame_rate / target_fps
    target_width = 1080
    aspect_ratio = frame_height / frame_width
    target_height = int(target_width * aspect_ratio)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = "output_video.mp4"
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, (target_width, target_height))

    stframe = st.empty()
    frame_counter = 0
    detection_interval = 1
    tracking = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to the target size
        frame = cv2.resize(frame, (target_width, target_height))

        if frame_counter % detection_interval == 0 or frame_counter <= 5:
            boxes, labels, scores = run_on_frame(models, frame, "1_1.jpg", iou_thr=0.7, skip_box_thr=0.0001, p=0.0001)
            labels = [int(label) for label in labels]

            detections = []
            for i in range(len(labels)):
                if scores[i] < 0.25:
                  continue
                boxes[i][0] = int(boxes[i][0] * target_width)
                boxes[i][1] = int(boxes[i][1] * target_height)
                boxes[i][2] = int(boxes[i][2] * target_width)
                boxes[i][3] = int(boxes[i][3] * target_height)

                boxes[i] = clip_bbox(boxes[i], target_width, target_height)

                boxes[i] = xyxy2ltwh(boxes[i])
                detections.append((boxes[i], scores[i], labels[i]))

            # Update Tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        deep_sort_boxes = []
        deep_sort_labels = []
        deep_sort_scores = []
        track_ids = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                tracking.pop(track.track_id, None)
                continue

            # filter out None values of confs
            if track.get_det_conf() is None:
                continue

            track_id = track.track_id
            track_class = track.get_det_class()
            track_conf = track.get_det_conf()

            if track_id not in tracking:
                tracking[track_id] = []
                tracking[track_id].append(track_class)
                tracking[track_id].append(track_conf)
            else:
                track_class = tracking[track_id][0]

            ltrb_box = track.to_ltrb()
            track_ids.append(track_id)
            deep_sort_boxes.append(ltrb_box)
            deep_sort_labels.append(track_class)
            deep_sort_scores.append(track_conf)

        # Plot bounding boxes on the frame
        frame_ = plot_bbox(frame, deep_sort_boxes, deep_sort_labels, deep_sort_scores, track_ids)

        # Save processed frame to video
        out.write(frame_)

        # Display the frame in Streamlit
        #stframe.image(frame_, channels="BGR")

        frame_counter += 1

    cap.release()
    out.release()  # Release VideoWriter
    st.success("Video processing completed!")

    # Provide download button for processed video
    with open(output_file, 'rb') as f:
        st.download_button("Download Processed Video", f, file_name="processed_video.mp4")

