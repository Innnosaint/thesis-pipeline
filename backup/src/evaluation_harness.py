"""
evaluation_harness.py
Objective 5 support: Computes rolling Precision, Recall, and False Positive
Rate as the real-time stream progresses, producing a live-performance curve
rather than a single static end-of-run score.
"""

import csv
from pathlib import Path


class RollingEvaluator:
    def __init__(self, rolling_window: int = 100):
        self.rolling_window = rolling_window
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.unlabeled = 0
        self.history = []  # list of dicts: {n, precision, recall, fpr}
        self.n_processed = 0

    def record(self, predicted_abnormal: bool, true_label):
        """
        true_label: True (anomaly), False (normal), or None (no ground truth
        available for this entry — excluded from confusion matrix but still
        counted in n_processed).
        """
        self.n_processed += 1

        if true_label is None:
            self.unlabeled += 1
            return

        if predicted_abnormal and true_label:
            self.tp += 1
        elif predicted_abnormal and not true_label:
            self.fp += 1
        elif not predicted_abnormal and true_label:
            self.fn += 1
        else:
            self.tn += 1

        if self.n_processed % self.rolling_window == 0:
            m = self.rolling_metrics()
            m["n"] = self.n_processed
            self.history.append(m)

    def rolling_metrics(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        fpr = self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0
        return {"precision": precision, "recall": recall, "fpr": fpr}

    def save_history(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["n", "precision", "recall", "fpr"])
            writer.writeheader()
            for row in self.history:
                writer.writerow(row)
