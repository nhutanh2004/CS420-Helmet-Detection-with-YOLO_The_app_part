import os
import json
from tqdm import tqdm


def count_samples_per_class(data):
    class_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for line in data:
        class_id = int(line[8])
        class_counts[class_id] += 1

    return class_counts, max(class_counts)


def find_max(results):
    classes_count, n_max_class = count_samples_per_class(results)
    return classes_count, n_max_class


def minority(p, results, n=9):
    """
    - Find the minority class and the threshold for the minority class
    @param p: min threshold
    @param results: predictions
    @param n: number of classes (=9)
    """
    classes_count, n_max_class = find_max(results)
    mean_samples = float(sum(classes_count) / n)  # mean samples per class
    alpha = float(
        n_max_class / mean_samples
    )  # mean samples per class / max samples in a class

    print(f"\n\tclasses count : {classes_count}")

    rare_classes = set()

    # find rare classes
    for index, each_class in enumerate(classes_count):
        if each_class < (n_max_class * alpha):
            rare_classes.add(index)

    min_thresh = 1

    # find minimum threshold
    for each_class_index in rare_classes:
        for sample in results:
            class_id = sample[8]
            score = sample[9]

            if class_id != each_class_index:
                continue
            if score < min_thresh:
                # print("\t\tupdating min_thresh, new threshold : ", score)
                min_thresh = score

    print(f"\n\tRare classes : {rare_classes}")
    print(f"\nMin_thresh = : {min_thresh}")

    return max(min_thresh, p), rare_classes


def minority_optimizer_func(results, p=0.001):
    number_of_classes = 9
    minority_score, rare_classes = minority(p, results, number_of_classes)

    print(f"Minority Score : {minority_score}\n\n")

    new_results = []
    for result in results:
        if result[-1] >= minority_score:
            new_results.append(result)

    # results[i] = [video_id, frame_id, x1, y1, x2, y2, img_w, img_h, label, score]
    return new_results


# OLD CODE (not using)
def count_samples_per_class_on_train(path="data/train/labels"):
    class_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    for file_name in tqdm(os.listdir(path), desc="Counting samples per class"):
        with open(os.path.join(path, file_name), "r") as f:
            for line in f:
                parts = line.strip().split(" ")
                class_id = int(float(parts[0]))
                class_counts[class_id] += 1

    return class_counts


def find_max_on_train():
    try:
        with open("config/constants.json", "r") as f:
            constants = json.load(f)
            classes_count = constants["classes_count"]
            if len(classes_count) != 9:
                raise Exception("Invalid classes count")
    except Exception as e:
        print(
            "\nCan't find classes count in constants.json, counting samples per class"
        )
        classes_count = count_samples_per_class_on_train()
        with open("config/constants.json", "w+") as f:
            json.dump({"classes_count": classes_count}, f)

    n_max_class = max(classes_count)
    print(f"\nclass counts : {classes_count}")
    return n_max_class, classes_count
    # max number of samples in a class, number of samples in each class
