import cv2
import matplotlib.pyplot as plt
import os

class_names = [
    "motorbike",
    "DHelmet",
    "DNoHelmet",
    "P1Helmet",
    "P1NoHelmet",
    "P2Helmet",
    "P2NoHelmet",
    "P0Helmet",
    "P0NoHelmet",
]

class_colors = [
    (255, 90, 90),
    (127, 245, 127),
    (90, 90, 255),
    (255, 255, 0),
    (120, 255, 255),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
]


# Hàm tính toán tỷ lệ font phù hợp
def get_optimal_font_scale(text, width, thickness=1):
    for scale in range(20, 0, -1):
        textSize = cv2.getTextSize(
            text,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=float(scale / 10),
            thickness=thickness,
        )[0]
        if textSize[0] <= width:
            return scale / 10

    return 0.1


def plot_bbox(
    image, boxes, labels, scores, names=class_names, class_colors=class_colors
):
    """
    Draw bounding boxes with labels and scores on the image.

    Parameters:
    - image: Image to draw boxes on.
    - boxes: List of bounding boxes (each box in [x1, y1, x2, y2]).
    - labels: List of class IDs for each box.
    - scores: List of confidence scores for each box.
    - color: Color to draw the boxes.
    - names: List of class names corresponding to class IDs.
    """
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        y1 += 3
        label = labels[i]
        score = scores[i]

        # Choose color for box
        color = class_colors[int(label)]

        # Create label text (class name and score)
        label_text = f"{names[label]}:{score*100:.0f}"
        max_width = 65 if x2 - x1 < 50 else x2 - x1
        max_width = min(max_width, 180)
        box_thickness = 1 if x2 - x1 < 100 else 2
        text_thickness = 1 if x2 - x1 < 80 else 2

        font_scale = get_optimal_font_scale(label_text, max_width, text_thickness)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=box_thickness)

        # Draw a rectangle behind text
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        top_left = (x1, y1 - text_height - 5)
        bottom_right = (x1 + text_width + 5, y1)
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

        # Put the label text on the image
        cv2.putText(
            image,
            label_text,
            (x1 + 3, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=text_thickness,
            lineType=cv2.LINE_AA,
        )

    return image


def visualize(image_path, predictions, plot=True):
    """
    Visualize predictions from 1 model on the same image.

    Parameters:
    - image_path: Path to the original image.
    - predictions: Predictions from the model (format: ["x1,y1,x2,y2,img_w,img_h,label,score\n"]).
    """
    # Load the image
    img = cv2.imread(image_path)
    img_model = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process model predictions
    boxes = []
    labels = []
    scores = []
    try:
        for line in predictions:
            x1, y1, x2, y2, w, h, label, score = map(float, line.strip().split(","))
            boxes.append([x1, y1, x2, y2])
            labels.append(int(label))
            scores.append(round(float(score), 2))
    except Exception as e:
        print(
            "The data is not in xyxywhls format. Trying label,x_center,y_center,bw,bh format"
        )
        for line in predictions:
            (
                label,
                x_center,
                y_center,
                bw,
                bh,
            ) = map(float, line.strip().split())
            x_center *= 1920
            y_center *= 1080
            bw *= 1920
            bh *= 1080

            x1 = int(x_center - bw / 2)
            y1 = int(y_center - bh / 2)
            x2 = int(x_center + bw / 2)
            y2 = int(y_center + bh / 2)
            score = 0
            boxes.append([x1, y1, x2, y2])
            labels.append(int(label))
            scores.append(round(float(score)))

    # Draw bounding boxes for model 1
    img_model = plot_bbox(img_model, boxes, labels, scores)

    # Display results side by side
    if plot:
        plt.figure(figsize=(12, 7))
        plt.axis("off")
        plt.title(f"{os.path.basename(image_path)}")
        plt.imshow(img_model)

        plt.show()
    return img_model


def compare(image_path, predictions_list):
    """
    Visualize multiple results on 1 image at once to compare

    Parameters:
    -image_path: path to the image
    -predictions_list: list of predictions
    """
    drawn_images = []
    for predictions in predictions_list:
        img = visualize(image_path, predictions, False)
        drawn_images.append(img)

    number_of_images = len(drawn_images)
    n_cols = 2
    n_rows = (number_of_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    for i, img in enumerate(drawn_images):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis("off")
        axes[row, col].imshow(img)
        # axes[row, col].set_title(f'Image {i+1}')

    plt.tight_layout()
    plt.show()
