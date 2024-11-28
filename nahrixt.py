import cv2
import torch
import numpy as np

import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class_names = [
    "motorbike",  # 0
    "DHelmet",  # 1
    "DNoHelmet",  # 2
    "P1Helmet",  # 3
    "P1NoHelmet",  # 4
    "P2Helmet",  # 5
    "P2NoHelmet",  # 6
    "P0Helmet",  # 7
    "P0NoHelmet",  # 8
]  # Danh sách tên class


minority_class = [3, 5, 6, 7, 8]
minority_conf = 0.005


conf_threshold = 0.5
tracking_class = None  # None: track all


tracker = DeepSort(
    max_age=20,
    n_init=3,
    max_cosine_distance=0.3,
    max_iou_distance=0.8,
    nms_max_overlap=0.7,
)
# tracker = DeepSort(
#     max_age=20,               # Faster updates
#     n_init=2,                 # Quick track confirmation
#     max_cosine_distance=0.3,  # Looser feature vector matching
#     max_iou_distance=0.8,     # Higher IoU for track association
#     nms_max_overlap=0.5,      # Stricter NMS threshold
# )

model = YOLO(f"weights\\v10m.pt")

colors = np.random.randint(0, 255, size=(len(class_names), 3))

tracks = []

frame_count = 0
detect_interval = 1  # Số frame giữa các lần YOLO detect

tracking = {}

# cap = cv2.VideoCapture("./10fps.mp4")
cap = cv2.VideoCapture(f"static\\short.mp4")

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

if model is None:
    print("Error: YOLO model could not be loaded.")
    exit()

names = os.listdir("nahrixt_out")
cv2_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
target_fps = 12
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
target_width = 640
aspect_ratio = frame_height / frame_width
target_height = int(target_width * aspect_ratio)
output_path = os.path.join("nahrixt_out", "predicting.mp4")
output = cv2.VideoWriter(
    output_path, cv2_fourcc, target_fps, (target_width, target_height)
)
# Tiến hành đọc từng frame từ video
while True:
    # Đọc
    ret, frame = cap.read()
    if not ret:
        # continue
        break
    # Đưa qua model để detect
    results = model(frame)
    # print("results : ", results)
    # Kiểm tra kết quả từ mô hình (results có thể là một danh sách)
    for result in results:
        # Truy cập vào các thuộc tính của từng kết quả
        boxes = result.boxes  # Đối tượng Boxes chứa các bounding box
        detections = []

        for det in boxes:
            # Lấy các giá trị bounding box
            x1, y1, x2, y2 = det.xyxy[0]  # Lấy tọa độ bounding box
            conf = det.conf[0]  # Độ tin cậy của đối tượng
            cls = det.cls[0]  # ID của lớp đối tượng

            # Kiểm tra độ tin cậy
            if cls in minority_class:
                if conf < minority_conf:
                    continue

            elif conf < conf_threshold:
                continue

            # if conf < conf_threshold:
            #     continue

            # Thêm phát hiện vào danh sách
            detect = [[int(x1), int(y1), int(x2 - x1), int(y2 - y1)], conf, int(cls)]
            detections.append(detect)

    # det_array = detections if len(detections) > 0 else np.empty((0, 5))
    # tracks = tracker.update_tracks(det_array, frame=frame)

    tracks = tracker.update_tracks(detections, frame=frame)

    # Vẽ lên màn hình các khung chữ nhật kèm ID
    for track in tracks:
        if track.is_confirmed():
            track_id = track.track_id

            # Lấy toạ độ, class_id để vẽ lên hình ảnh
            ltrb = track.to_ltrb()
            class_id = track.get_det_class()
            x1, y1, x2, y2 = map(int, ltrb)
            color = colors[class_id]
            B, G, R = map(int, color)

            if track_id not in tracking:
                tracking[track_id] = []
                tracking[track_id].append(class_id)
                tracking[track_id].append(B)
                tracking[track_id].append(G)
                tracking[track_id].append(R)
            else:
                class_id = tracking[track_id][0]
                B, G, R = tracking[track_id][1:]

            # label = "{}-{}".format(class_names[class_id], track_id)
            label = "{}-{}".format(class_names[class_id], track_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(
                frame, (x1 - 1, y1 - 20), (x1 + len(label) * 12, y1), (B, G, R), -1
            )
            cv2.putText(
                frame,
                label,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

    output.write(frame)
    frame_count += 1
    # Show hình ảnh lên màn hình
    resize_frame = cv2.resize(frame, (960, 720))
    cv2.imshow("OT", resize_frame)
    cv2.waitKey(3)
    # Bấm Q thì thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
output.release()
compressed_output_path = os.path.join("nahrixt_out", f"{len(names) + 1}.mp4")
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

import subprocess

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


cv2.destroyAllWindows()
