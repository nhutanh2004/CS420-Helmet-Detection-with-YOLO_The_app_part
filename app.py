from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    send_from_directory,
)
from ultralytics import YOLO
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import subprocess
import sys

sys.path.append("backend/source")

from backend.source.test import run_on_frame, create_models
from backend.source.utils.visualize import plot_bbox

from deep_sort_realtime.deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime import deep_sort_realtime


app = Flask(__name__)

# Ensure static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Ensure weights directory exists
if not os.path.exists("weights"):
    os.makedirs("weights")


# Ensure boxs don't go outside frame
def clip_bbox(bbox, frame_width, frame_height):
    x1 = max(0, min(bbox[0], frame_width - 1))
    y1 = max(0, min(bbox[1], frame_height - 1))
    x2 = max(0, min(bbox[2], frame_width - 1))
    y2 = max(0, min(bbox[3], frame_height - 1))
    return [x1, y1, x2, y2]


# Create a list of model weights
def get_model_weights():
    weights_dir = "weights"
    return [
        os.path.join(weights_dir, model_name)
        for model_name in os.listdir(weights_dir)
        if model_name.endswith(".pt")
    ]


# convert xyxy to ltwh
def xyxy2ltwh(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [x1, y1, w, h]


# Create models using the provided weights
model_weights_list = get_model_weights()
models, model_names = create_models(model_weights_list)


# Initialize DeepSORT
tracker = DeepSort(
    max_age=20,
    n_init=3,
    max_cosine_distance=0.35,
    max_iou_distance=0.8,
    nms_max_overlap=0.7,
)


@app.route("/")
def upload_video():
    return render_template("video.html")


@app.route("/process_video", methods=["POST"])
def process_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    filename = secure_filename(file.filename)
    file_path = os.path.join("static", filename)
    file.save(file_path)

    # Get parameters from the request
    iou_thr = float(request.form["iou_thr"])
    skip_box_thr = float(request.form["skip_box_thr"])
    p = float(request.form["p"])

    # Open the video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return jsonify({"error": "Unable to open video file"})

    # Get frame dimensions and calculate target size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = 20  # Desired frames per second for processing

    if target_fps > frame_rate:
        target_fps = frame_rate

    frame_interval = frame_rate // target_fps  # Interval to skip frames

    target_width = 1080
    aspect_ratio = frame_height / frame_width
    target_height = int(target_width * aspect_ratio)

    output_path = os.path.join("static", f"predicted_{filename}")
    cv2_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(
        output_path, cv2_fourcc, target_fps, (target_width, target_height)
    )

    frame_counter = 0  # Initialize frame counter

    # trick from nahrixt.py
    tracking = {}
    detection_interval = 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to meet the target FPS
        # if frame_counter % frame_interval != 0:
        #     frame_counter += 1
        #     continue

        # Resize frame to the target size
        frame = cv2.resize(frame, (target_width, target_height))

        # Skip detection for some frames
        if frame_counter % detection_interval == 0 or frame_counter <= 5:

            # Convert the frame to RGB
            img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run the models on the frame (bbox is in xyxy format)
            boxes, labels, scores = run_on_frame(
                models, img_array, "1_1.jpg", iou_thr, skip_box_thr, p
            )
            # Convert the labels to integers
            labels = [int(label) for label in labels]
            # Truncate virtual boxes and denormalize real boxes
            detections = []
            for i in range(len(labels)):
                if scores[i] < 0.25:
                    continue
                # Denormalize the boxes
                boxes[i][0] = int(boxes[i][0] * target_width)
                boxes[i][1] = int(boxes[i][1] * target_height)
                boxes[i][2] = int(boxes[i][2] * target_width)
                boxes[i][3] = int(boxes[i][3] * target_height)
                # Clip the boxes to the frame
                boxes[i] = clip_bbox(boxes[i], target_width, target_height)
                # Convert xyxy to ltwh
                boxes[i] = xyxy2ltwh(boxes[i])
                detections.append((boxes[i], scores[i], labels[i]))
            # Update tracker
            tracks = tracker.update_tracks(detections, frame=frame)
        else:
            tracks = tracker.update_tracks([], frame=frame)
            if len(tracks) == 0:
                print("\tNo tracks found")

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
                # if tracking[track_id][1] < track_conf:
                #     tracking[track_id][0] = track_class
                #     tracking[track_id][1] = track_conf
                track_class = tracking[track_id][0]

            ltrb_box = track.to_ltrb()
            track_ids.append(track.track_id)
            deep_sort_boxes.append(ltrb_box)
            deep_sort_labels.append(track_class)
            deep_sort_scores.append(track.get_det_conf())

        print(
            f"deep_sort_boxes: {len(deep_sort_boxes)}, {len(deep_sort_labels)}, {len(deep_sort_scores)}"
        )
        print(f"deep_sort_scores: {deep_sort_scores}")
        print(f"deep_sort_labels: {deep_sort_labels}\n\n")

        # Draw bounding boxes on the frame
        drew_frame = plot_bbox(
            frame,
            deep_sort_boxes,
            deep_sort_labels,
            deep_sort_scores,
            track_ids,
        )

        # Write the processed frame to the output video
        output.write(drew_frame)
        frame_counter += 1  # Increment frame counter

        # Show hình ảnh lên màn hình
        resize_frame = cv2.resize(drew_frame, (960, 720))
        cv2.imshow("OT", resize_frame)
        cv2.waitKey(3)
        # Bấm Q thì thoát
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    output.release()

    # Compress the video using ffmpeg command line and capture stderr
    counter = 0
    compressed_output_path = os.path.join("static", f"{filename.split('.')[0]}.mp4")
    while os.path.exists(compressed_output_path):
        counter += 1
        compressed_output_path = os.path.join(
            "static", f"{filename.split('.')[0]}_{counter}.mp4"
        )

    ffmpeg_cmd = [
        "./ffmpeg.exe",
        "-i",
        output_path,
        "-vcodec",
        "h264",
        "-acodec",
        "aac",
        "-strict",
        "-2",
        compressed_output_path,
    ]
    # Debug for some time the compressed output does not save properly
    try:
        result = subprocess.run(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
        else:
            print(f"Compressed video created successfully at: {compressed_output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    # Delete the intermediate processed video file
    if os.path.exists(output_path):
        os.remove(output_path)

    # # Delete the original video file
    # if os.path.exists(file_path):
    #     os.remove(file_path)

    return jsonify(
        {
            "original_video_url": f"/static/{filename}",
            "processed_video_url": f"/{compressed_output_path}",
        }
    )


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    return send_from_directory("static", filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
