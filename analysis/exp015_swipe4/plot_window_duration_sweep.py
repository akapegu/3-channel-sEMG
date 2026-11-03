"""Plot event/gesture accuracy and tilt-control errors from duration sweep."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPORT = HERE / "session1_window_duration_sweep.json"
OUTPUT = HERE / "session1_window_duration_sweep.png"


def main():
    data = json.loads(REPORT.read_text())["models"]
    families = ["EXTRA_TREES_EMG", "EXTRA_TREES_FUSION", "SVM_RBF_EMG", "SVM_RBF_FUSION"]
    labels = {
        "EXTRA_TREES_EMG": "Extra Trees · EMG",
        "EXTRA_TREES_FUSION": "Extra Trees · EMG + accel",
        "SVM_RBF_EMG": "RBF SVM · EMG",
        "SVM_RBF_FUSION": "RBF SVM · EMG + accel",
    }
    durations = [100, 150, 200, 250, 300, 400]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = [
        ("Correct cue intervals (%)", "event_accuracy"),
        ("Correct gesture intervals (%)", "gesture_event_accuracy"),
        ("Tilt-only controls called commands (%)", "tilt_only_false_command_rate"),
    ]
    for ax, (title, field) in zip(axes, panels):
        for model, line_color in zip(families, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]):
            values = [100 * data[f"{model}_{d}ms"]["pooled_out_of_fold"][field] for d in durations]
            ax.plot(durations, values, marker="o", linewidth=2, label=labels[model], color=line_color)
        ax.set_title(title)
        ax.set_xlabel("Window duration (ms)")
        ax.set_xticks(durations)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Percent")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Session 1 grouped CV — fixed 20 ms update hop\nExploratory within one wearing; event-vote scores do not show live latency")
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
