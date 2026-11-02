"""Plot window-level and per-cue confusion from the one locked test report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT / "model_v1" / "locked_test_metrics.json").read_text())
labels = report["metrics"]["confusion_matrix_labels"]
window = np.asarray(report["metrics"]["confusion_matrix_rows_true_columns_pred"], dtype=float)
cue = np.asarray(report["metrics"]["active_cue_confusion_rows_true_columns_pred"], dtype=float)[1:, :]
cue_labels = labels[1:]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for ax, matrix, true_names, predicted_names, title in (
    (axes[0], window, labels, labels, "Overlapping 200 ms windows"),
    (axes[1], cue, cue_labels, labels, "Majority vote per prompted swipe (n=32)"),
):
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(predicted_names)), predicted_names, rotation=35, ha="right")
    ax.set_yticks(range(len(true_names)), true_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{normalized[i,j]:.0%}\n({int(matrix[i,j])})",
                    ha="center", va="center",
                    color="white" if normalized[i,j] > .55 else "#172033", fontsize=9)
fig.colorbar(image, ax=axes, shrink=.8, label="Fraction within true class")
fig.suptitle("Four-swipe baseline — locked same-session test block")
out = ROOT / "model_v1" / "locked_test_confusion.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=170)
print(out)
