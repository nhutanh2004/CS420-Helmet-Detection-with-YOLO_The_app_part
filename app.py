from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
import cv2
import os
from werkzeug.utils import secure_filename
import subprocess
import sys

sys.path.append("backend/source")

from backend.source.test import run_on_frame, create_models
from backend.source.utils.visualize import plot_bbox
from deep_sort_realtime.deepsort_tracker import DeepSort
# import logging
from voting_system import VotingSystem
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure logging
# logging.basicConfig(
#     filename='debug.txt',
#     level=logging.DEBUG,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )

# Ensure directories exist
for directory in ["static", "weights", app.config['UPLOAD_FOLDER']]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Utility functions
def clip_bbox(bbox, frame_width, frame_height):
    x1 = max(0, min(bbox[0], frame_width - 1))
    y1 = max(0, min(bbox[1], frame_height - 1))
    x2 = max(0, min(bbox[2], frame_width - 1))
    y2 = max(0, min(bbox[3], frame_height - 1))
    return [x1, y1, x2, y2]

def ltwh2xyxy(box):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    return [x1, y1, x2, y2]

def xyxy2ltwh(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return [x1, y1, w, h]

# Load model weights
def get_model_weights():
    weights_dir = "weights"
    return [
        os.path.join(weights_dir, model_name)
        for model_name in os.listdir(weights_dir)
        if model_name.endswith(".pt")
    ]

# Create models using the provided weights
model_weights_list = get_model_weights()
models, model_names = create_models(model_weights_list)
voting_system = VotingSystem(frame_window=10)
@app.route("/") 
def index(): 
    return render_template("home.html") 
@app.route("/video") 
def video_page(): 
    return render_template("video.html") 
@app.route("/live_stream") 
def live_stream_page(): 
    return render_template("video_live.html")

@app.route("/upload_video", methods=["POST"])
def upload_video_route():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Return the file path to be used in live streaming
    return jsonify({"success": True, "file_path": file_path})

@app.route("/process_live_stream", methods=["GET"])
def process_live_stream():
    file_path = request.args.get('file')
    if not file_path:
        return "Error: No file path provided", 400

    # Get parameters from the request
    try:
        iou_thr = float(request.args.get("iou_thr"))
        skip_box_thr = float(request.args.get("skip_box_thr"))
        p = float(request.args.get("p"))
        max_age = int(request.args.get("max_age"))
        n_init = int(request.args.get("n_init"))
        max_cosine_distance = float(request.args.get("max_cosine_distance"))
        nms_max_overlap = float(request.args.get("nms_max_overlap"))
        frame_window = int(request.args.get("frame_window"))
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    # Initialize DeepSORT and Voting System for live stream
    tracker = DeepSort(max_age=max_age, nms_max_overlap=nms_max_overlap, n_init=n_init, max_cosine_distance=max_cosine_distance)
    voting_system = VotingSystem(frame_window=frame_window)

    def gen():
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Perform detection with YOLO
            img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, labels, scores = run_on_frame(models, img_array, "1_1.jpg", iou_thr, skip_box_thr, p)
            labels = [int(label) for label in labels]

            detections = []
            for i in range(len(labels)):
                if scores[i] > 0.001:
                    boxes[i][0] = int(boxes[i][0] * frame.shape[1])
                    boxes[i][1] = int(boxes[i][1] * frame.shape[0])
                    boxes[i][2] = int(boxes[i][2] * frame.shape[1])
                    boxes[i][3] = int(boxes[i][3] * frame.shape[0])
                    boxes[i] = xyxy2ltwh(boxes[i])
                    detections.append((boxes[i], scores[i], labels[i]))

            # Update DeepSORT tracker with the latest detections
            tracks = tracker.update_tracks(detections, frame=frame)

            # Process each frame
            for track in tracks:
                if track.is_confirmed() and track.time_since_update <= 1:
                    bbox = track.to_ltrb()
                    bbox = clip_bbox(bbox, frame.shape[1], frame.shape[0])

                    track_id = track.track_id
                    det_label = track.get_det_class()
                    det_score = track.get_det_conf()

                    # Update the voting system with the latest detection data
                    voting_system.update_track(track_id, det_label, det_score)

                    # Get the voted label and average score
                    voted_label, avg_score = voting_system.get_voted_label_and_score(track_id)

                    # Draw the bounding box with the stored label and score
                    frame = plot_bbox(frame, [bbox], [voted_label], [avg_score])

            # Encode frame to JPEG format
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame.")
                continue

            # Convert encoded frame to bytes and yield it in the response
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        cap.release()

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/process_video", methods=["POST"])
def process_video():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join("static", filename)
    file.save(file_path)

    # Get parameters from the request
    try:
        iou_thr = float(request.form["iou_thr"])
        skip_box_thr = float(request.form["skip_box_thr"])
        p = float(request.form["p"])
        max_age = int(request.form["max_age"])
        n_init = int(request.form["n_init"])
        max_cosine_distance = float(request.form["max_cosine_distance"])
        max_iou_distance = float(request.form["max_iou_distance"])
        nms_max_overlap = float(request.form["nms_max_overlap"])
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    # Initialize DeepSORT
    tracker = DeepSort(max_age=max_age, nms_max_overlap=nms_max_overlap, n_init=n_init, max_cosine_distance=max_cosine_distance, max_iou_distance=max_iou_distance, nn_budget=None, override_track_class=None)

    # Open the video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return jsonify({"error": "Unable to open video file"})

    # Get frame dimensions and calculate target size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = 30  # Desired frames per second for processing

    if target_fps > frame_rate:
        target_fps = frame_rate

    frame_interval = frame_rate // target_fps  # Interval to skip frames

    target_width = 640
    aspect_ratio = frame_height / frame_width
    target_height = int(target_width * aspect_ratio)

    output_path = os.path.join("static", f"predicted_{filename}")
    cv2_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(output_path, cv2_fourcc, target_fps, (target_width, target_height))

    frame_counter = 0  # Initialize frame counter
    detection_interval = 1  # Detect every frame

    track_info = {}  # Initialize track info dictionary

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to meet the target FPS
        if frame_counter % frame_interval != 0:
            frame_counter += 1
            continue

        # Resize frame to the target size
        frame = cv2.resize(frame, (target_width, target_height))

        detections = []
        if frame_counter % detection_interval == 0:
            # Perform detection with YOLO
            img_array = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, labels, scores = run_on_frame(models, img_array, "1_1.jpg", iou_thr, skip_box_thr, p)
            labels = [int(label) for label in labels]

            for i in range(len(labels)):
                if scores[i] > 0.001:
                    boxes[i][0] = int(boxes[i][0] * target_width)
                    boxes[i][1] = int(boxes[i][1] * target_height)
                    boxes[i][2] = int(boxes[i][2] * target_width)
                    boxes[i][3] = int(boxes[i][3] * target_height)
                    boxes[i] = xyxy2ltwh(boxes[i])
                    detections.append((boxes[i], scores[i], labels[i]))
                    # Update VotingSystem
                    voting_system.update_track(i, labels[i], scores[i])

        # Update DeepSORT tracker with the latest detections
        tracks = tracker.update_tracks(detections, frame=frame)

        # Process each frame
        for track in tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_ltrb()
                bbox = clip_bbox(bbox, frame_width, frame_height)

                track_id = track.track_id
                # Get voted label and score from VotingSystem
                voted_label, voted_score = voting_system.get_voted_label_and_score(track_id)

                # Draw the bounding box with the voted label and score
                frame = plot_bbox(frame, [bbox], [voted_label], [voted_score])

        output.write(frame)  # Write the processed frame to the output video
        frame_counter += 1  # Increment frame counter

    cap.release()
    output.release()

    # Compress the video using ffmpeg command line and capture stderr
    counter = 0
    compressed_output_path = os.path.join(
        "static", f"{filename.split('.')[0]}.mp4"
    )
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

    # Return the URLs for the original and processed video
    return jsonify({
        "original_video_url": f"/static/{filename}",
        "processed_video_url": f"/{compressed_output_path}",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
