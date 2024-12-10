
import numpy as np
from collections import defaultdict, Counter

class VotingSystem:
    def __init__(self, frame_window=10):
        self.frame_window = frame_window
        self.track_history = defaultdict(lambda: {"labels": [], "scores": [], "initial_label": None, "initial_score": None})

    def update_track(self, track_id, label, score):
        history = self.track_history[track_id]

        if not history["labels"]:
            history["initial_label"] = label
            history["initial_score"] = score

        if len(history["labels"]) >= self.frame_window:
            history["labels"].pop(0)
            history["scores"].pop(0)

        history["labels"].append(label)
        history["scores"].append(score)

    def get_voted_label_and_score(self, track_id):
        history = self.track_history[track_id]
        label_list = history["labels"]
        score_list = history["scores"]

        if len(label_list) < self.frame_window or len(score_list) < self.frame_window:
            return history["initial_label"], history["initial_score"]

        # Filter out None values from the score list
        valid_scores = [score for score in score_list if score is not None]

        if not valid_scores:
            return history["initial_label"], history["initial_score"]

        # Determine the most frequent label
        most_common_label = Counter(label_list).most_common(1)[0][0]

        # Calculate the average score
        average_score = np.mean(valid_scores)

        return most_common_label, average_score

    def clear_history(self):
        self.track_history.clear()

# Example usage
if __name__ == "__main__":
    voting_system = VotingSystem(frame_window=10)

    # Simulate updates with track ID 1
    for _ in range(3):  # Simulate 3 frames (less than frame window)
        voting_system.update_track(1, 'person', 0.9)

    # Get the voted label and average score with insufficient data
    label, score = voting_system.get_voted_label_and_score(1)
    print(f"Voted Label: {label}, Average Score: {score}")  # Should use initial values ('person', 0.9)
