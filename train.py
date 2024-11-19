# this file is use to train models

# Run this file by : python train.py --model_path path/to/model.pt --epochs 30


import argparse
from ultralytics import YOLO
import os

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Train YOLO model")

    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs for training"
    )  # New argument for epochs
    args = parser.parse_args()

    # Load the model from the provided path
    model = YOLO(args.model_path)
    print("Imported model:", args.model_path)

    # Start the training process
    print("\n\n********** START TRAINING *************\n\n")

    results = model.train(
        data="../config/coco.yaml",
        epochs=args.epochs,  # Use the epochs argument
        imgsz=640,
        save_dir="../weights",
    )

    # Save the best model to best.pt
    best_model_path = os.path.join("../weights", "best.pt")

    model.save(best_model_path)

    print(f"Best model saved to: {best_model_path}")

    print("\n\n\n\nDONE TRAINING\n\n")
