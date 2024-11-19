from visualize import *
import os


label_name = "1.mp4_54.txt"
image_name = "1.mp4_54.jpg"

path = "data/small_val"
label = os.path.join(path, "labels", label_name)
image = os.path.join(path, "images", image_name)

print(os.path.exists(label))
print(os.path.exists(image))

gt = []
count = 0
with open(label, "r") as file:
    for line in file:
        count += 1
        print(line)
        gt.append(line)

print(f"number of boxes: {count}")

visualize(image, gt, True)
