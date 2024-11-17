import cv2
import matplotlib.pyplot as plt
import os


def print_():
    print("hello world from visualize")


COLORS = [
    (255, 179, 0),
    (128, 62, 117),
    (255, 104, 0),
    (166, 189, 215),
    (193, 0, 32),
    (206, 162, 98),
    (129, 112, 102),
    (0, 125, 52),
    (246, 118, 142),
    (0, 83, 138),
    (255, 122, 92),
    (83, 55, 122),
    (255, 142, 0),
    (179, 40, 81),
    (244, 200, 0),
    (127, 24, 13),
    (147, 170, 0),
    (89, 51, 21),
    (241, 58, 19),
    (35, 44, 22),
]

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
    (0, 255, 255),  # Xanh lơ
    (0, 255, 0),  # Xanh lá
    (0, 0, 255),  # Xanh dương
    (255, 255, 0),  # Vàng
    (255, 0, 255),  # Hồng
    (255, 0, 0),  # Đỏ
    (128, 0, 128),  # Tím
    (0, 128, 128),  # Xanh biển
    (128, 128, 0),  # Xanh rêu
]


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
        label = labels[i]
        score = scores[i]

        # Choose color for box
        color = class_colors[int(label)]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Create label text (class name and score)
        label_text = f"{names[label]}: {score:.2f}"

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        )

        # Draw a rectangle behind text for background blur
        rect_start = (x1, y1 - text_height - 10)
        rect_end = (x1 + text_width, y1)
        sub_img = image[rect_start[1] : rect_end[1], rect_start[0] : rect_end[0]]

        # Apply Gaussian blur for a blurred rectangle effect
        if sub_img.shape[0] > 0 and sub_img.shape[1] > 0:
            blurred_rect = cv2.GaussianBlur(sub_img, (5, 5), 0)
            image[rect_start[1] : rect_end[1], rect_start[0] : rect_end[0]] = (
                blurred_rect
            )

        # Draw a semi-transparent rectangle as text background
        overlay = image.copy()
        cv2.rectangle(overlay, rect_start, rect_end, (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Put the label text on the image
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
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
