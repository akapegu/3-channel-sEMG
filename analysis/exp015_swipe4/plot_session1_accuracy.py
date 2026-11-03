"""Plot event-level confusion matrices for Session 1 baseline checks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LABELS = ["REST*", "DOWN", "LEFT", "RIGHT", "TOP"]


def main():
    transfer = json.loads((HERE / "session1_model_comparison.json").read_text())
    cv = json.loads((HERE / "session1_within_session_grouped_cv.json").read_text())
    specs = [
        ("Previous frozen model\nLDA + fusion", transfer["models"]["LOCKED_LDA_FUSION"]),
        ("Within-sitting grouped CV\nSVM + EMG", cv["models"]["SVM_EMG"]["pooled_out_of_fold"]),
        ("Within-sitting grouped CV\nSVM + fusion", cv["models"]["SVM_FUSION"]["pooled_out_of_fold"]),
        ("Within-sitting grouped CV\nSVM + accelerometer only", cv["models"]["SVM_ACCEL_ONLY"]["pooled_out_of_fold"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    last_image = None
    for ax, (title, metrics) in zip(axes.flat, specs):
        cm = np.asarray(metrics["event_confusion_true_rows_pred_columns"], dtype=int)
        row_totals = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0)
        last_image = ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
        ax.set_title(title + f"\nEvent accuracy {metrics['event_accuracy']:.1%}", fontsize=12)
        ax.set_xticks(range(5), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(5), LABELS)
        ax.set_xlabel("Predicted cue")
        ax.set_ylabel("True cue")
        for i in range(5):
            for j in range(5):
                color = "white" if norm[i, j] >= 0.55 else "black"
                ax.text(j, i, f"{cm[i,j]}\n{norm[i,j]:.0%}", ha="center", va="center",
                        color=color, fontsize=9)
    fig.suptitle("Session 1 event-level errors\n*REST rows are 40 relaxed-finger tilt-only controls", fontsize=15)
    fig.colorbar(last_image, ax=list(axes.flat), shrink=0.82, label="Share of each true class")
    out = HERE / "session1_accuracy_confusion.png"
    fig.savefig(out, dpi=180)
    print(out)


if __name__ == "__main__":
    main()
