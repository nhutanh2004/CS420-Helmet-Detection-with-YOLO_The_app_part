from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from ultralytics import YOLO
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import subprocess
from wbf import *
from visualize import *
from minority_optimizer import *
from virtual_expander import *
from test import *
app = Flask(__name__)

# Ensure static directory exists
if not os.path.exists('static'):
    os.makedirs('static')

# Ensure weights directory exists
if not os.path.exists('weights'):
    os.makedirs('weights')

# Create a list of model weights
def get_model_weights():
    weights_dir = 'weights'
    return [os.path.join(weights_dir, model_name) for model_name in os.listdir(weights_dir) if model_name.endswith('.pt')]

# Create models using the provided weights
model_weights_list = get_model_weights()
models, model_names = create_models(model_weights_list)

# Define class names
class_names = ['motorbike', 'DHelmet', 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet', 'P0Helmet', 'P0NoHelmet']
class_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128)]

@app.route('/')
def upload_video():
    return render_template('video.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    filename = secure_filename(file.filename)
    file_path = os.path.join('static', filename)
    file.save(file_path)
    
    # Open the video file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return jsonify({"error": "Unable to open video file"})

    # Get frame dimensions and calculate target size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    target_fps = 10  # Desired frames per second for processing
    
    if target_fps > frame_rate:
        target_fps=frame_rate
    
    frame_interval = frame_rate // target_fps  # Interval to skip frames
    
    target_width = 640
    aspect_ratio = frame_height / frame_width
    target_height = int(target_width * aspect_ratio)
    
    output_path = os.path.join('static', f"predicted_{filename}")
    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(output_path, cv2_fourcc, target_fps, (target_width, target_height))

    frame_counter = 0  # Initialize frame counter

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

        # Save the resized frame temporarily for detection
        temp_image_path = os.path.join('static', '1.mp4_1.jpg')
        cv2.imwrite(temp_image_path, frame)

        # Run predictions on the single frame using run_on_single_image
        p = 0.001  # Min confidence threshold for rare classes
        iou_thr = 0.5  # IOU threshold for WBF
        sbthr = 0.00001  # Skip box threshold for WBF
        plot = False  # Flag to plot the results, set to False for video processing

        results = run_on_single_image(model_weights_list, temp_image_path, p, iou_thr, sbthr, plot)

        # Debugging statement to understand the structure of results
        # print(f"Results for frame {frame_counter}: {results}")

        # Initialize lists for bounding boxes, labels, and scores
        boxes, labels, scores = [], [], []

        # Extract the bounding boxes, labels, and scores from results
        for preds in results:
            if isinstance(preds, list):
                # Directly use preds since it's a single prediction list
                x1, y1, x2, y2 = [preds[2], preds[3], preds[4], preds[5]]
                label = int(preds[8])
                score = preds[9]
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
                scores.append(score)
                #print(f"Labels: {labels}")  # Debug print to see the labels being appended


        
        # Draw the bounding boxes on the frame using plot_bbox function
        frame = plot_bbox(frame, boxes, labels, scores, names=class_names, class_colors=class_colors)

        # Write the processed frame to the output video
        output.write(frame)
        frame_counter += 1  # Increment frame counter

    cap.release()
    output.release()

    # Compress the video using ffmpeg command line and capture stderr 
    compressed_output_path = os.path.join('static', f"compressed_{filename}")
    ffmpeg_cmd = ['./ffmpeg.exe',
                  '-i', output_path,
                  '-vcodec', 'h264',
                  '-acodec', 'aac',
                  '-strict', '-2',
                  compressed_output_path
                  ]
    # Debug for some time the compressed output does not save properly
    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
        else:
            print(f"Compressed video created successfully at: {compressed_output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
    # Delete the intermediate processed video file
    if os.path.exists(output_path):
       os.remove(output_path)
    
    # Delete the original video file
    if os.path.exists(file_path):
        os.remove(file_path)
        
    return jsonify({"video_url": f"/static/compressed_{filename}"})

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('static', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
