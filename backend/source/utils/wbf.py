import os
import cv2
from ensemble_boxes import *
import sys
import torch
import warnings


# FUNCTION TO DETECT AND FUSE
def detect_image(image_path, model):
    """
    Detect 1 image with specified model
    """
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    results = model.predict(
        source=img, save=False, stream=True, batch=8, conf=0.00001, device=device
    )

    lines = []
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0]
            score = box.conf[0].item()
            label = int(box.cls[0].item())
            # NOT normalized xyxy and h,w are image_h and image_w
            lines.append(
                f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{img_w},{img_h},{label},{score}\n"
            )
    return lines


def fuse(
    models_names_list,
    test_path,
    predictions,
    single_image=False,
    iou_thr=0.5,
    skip_box_thr=0.001,
):
    """
    - Fuse results from multiple models of all images or single image

    - Params: (model_names_list, test_path, predicitons of all models, iou_thres, skip_box_thr)
    """
    results = []

    # Tắt toàn bộ các cảnh báo
    warnings.filterwarnings("ignore")

    # Hoặc tắt cảnh báo cụ thể
    warnings.filterwarnings("ignore", message="Zero area box skipped*")

    print("\n\tFusing results with iou_thr:", iou_thr, "skip_box_thr:", skip_box_thr)

    # Iterate through all images
    if not single_image:
        for image_name in os.listdir(test_path):

            video_id, frame_id = image_name.split(".mp4_")
            frame_id = frame_id.split(".jpg")[0]

            img_boxes = []
            img_labels = []
            img_scores = []
            i_h, i_w = cv2.imread(os.path.join(test_path, image_name)).shape[
                :2
            ]  # store the image size

            for model in models_names_list:
                model_prediction = predictions[model].get(image_name, [])
                for line in model_prediction:
                    # Extract information from the line and normalize bbox
                    try:
                        x1, y1, x2, y2, img_w, img_h, label, score = line.strip().split(
                            ","
                        )
                        img_w = float(img_w)
                        img_h = float(img_h)
                        box = (
                            float(x1) / img_w,
                            float(y1) / img_h,
                            float(x2) / img_w,
                            float(y2) / img_h,
                        )
                        img_boxes.append(box)
                        img_labels.append(int(label))
                        img_scores.append(float(score))
                    except Exception as e:
                        print(
                            f"Error processing line {line} in image {image_name}: {e}"
                        )
                        continue

            # fuse boxes
            if img_boxes:
                # weighted_boxes_fusion expects lists of lists
                boxes, scores, labels = weighted_boxes_fusion(
                    [img_boxes],  # Make it a list of lists
                    [img_scores],  # Make it a list of lists
                    [img_labels],  # Make it a list of lists
                    iou_thr=iou_thr,
                    skip_box_thr=skip_box_thr,
                )

                # save results with de-normalized bounding box
                for i in range(len(boxes)):
                    results.append(
                        [
                            int(video_id),
                            int(frame_id),
                            boxes[i][0] * i_w,
                            boxes[i][1] * i_h,
                            boxes[i][2] * i_w,
                            boxes[i][3] * i_h,
                            i_w,
                            i_h,
                            labels[i],
                            scores[i],
                        ]
                    )
    if single_image:
        image_name = os.path.basename(test_path)

        video_id, frame_id = image_name.split(".mp4_")
        frame_id = frame_id.split(".jpg")[0]

        img_boxes = []
        img_labels = []
        img_scores = []

        i_h, i_w = cv2.imread(test_path).shape[:2]  # store the image size

        for model in models_names_list:
            model_prediction = predictions[model][image_name]
            for line in model_prediction:
                # Extract information from the line and normalize bbox
                try:
                    x1, y1, x2, y2, img_w, img_h, label, score = line.strip().split(",")
                    img_w = float(img_w)
                    img_h = float(img_h)
                    box = (
                        float(x1) / img_w,
                        float(y1) / img_h,
                        float(x2) / img_w,
                        float(y2) / img_h,
                    )
                    img_boxes.append(box)
                    img_labels.append(int(label))
                    img_scores.append(float(score))
                except Exception as e:
                    print(f"Error processing line {line} in image: {e}")
                    continue
        if img_boxes:
            # weighted_boxes_fusion expects lists of lists
            boxes, scores, labels = weighted_boxes_fusion(
                [img_boxes],  # Make it a list of lists
                [img_scores],  # Make it a list of lists
                [img_labels],  # Make it a list of lists
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
            # save results with de-normalized bounding box
            # save results with de-normalized bounding box
            for i in range(len(boxes)):
                results.append(
                    [
                        int(video_id),
                        int(frame_id),
                        boxes[i][0] * i_w,
                        boxes[i][1] * i_h,
                        boxes[i][2] * i_w,
                        boxes[i][3] * i_h,
                        i_w,
                        i_h,
                        labels[i],
                        scores[i],
                    ]
                )

    # results[i] = [video_id, frame_id, x1, y1, x2, y2, img_w, img_h, label, score]
    return results


def fuse_frame(
    all_boxes, all_labels, all_scores, iou_thr, skip_box_thr, frame_name, i_w, i_h
):
    """
    Fuse results from multiple models of a single frame, format output to match MO, VE
    """
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels, iou_thr=iou_thr, skip_box_thr=skip_box_thr
    )

    # format output: attach vid, fid, denormalize bbox
    # frame_name = "vid_fid.jpg"
    vid, fid = frame_name.split(".jpg")[0].split("_")
    results = []

    for i in range(len(fused_boxes)):
        results.append(
            [
                int(vid),
                int(fid),
                fused_boxes[i][0],
                fused_boxes[i][1],
                fused_boxes[i][2],
                fused_boxes[i][3],
                i_w,
                i_h,
                fused_labels[i],
                fused_scores[i],
            ]
        )
    # results[i] = [video_id, frame_id, x1, y1, x2, y2, img_w, img_h, label, score]

    return results
