"""Compare training and validation feature distributions only (never test)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "supplier_toolkit/adapted/training_data"
OUT = Path(__file__).resolve().parent
CLASSES = ("rest", "pinch", "flexion", "extension")
PARTS = ("train", "validation")
WINDOW = 500
STRIDE = 250


def load_intervals(path: Path, n: int):
    review = path.with_suffix(".segments_review.json")
    cue = path.with_suffix(".beep.json")
    if review.exists():
        intervals = json.loads(review.read_text())["segments"]
    elif cue.exists():
        intervals = json.loads(cue.read_text()).get("segments", [])
    else:
        raise ValueError(f"No cue intervals available for {path}")
    return [(max(0, int(a)), min(n, int(b))) for a, b in intervals if int(b) > int(a)]


def iter_windows(data, intervals):
    for start, end in intervals:
        i = start
        while i + WINDOW <= end:
            yield data[i:i + WINDOW]
            i += STRIDE


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {"count": int(a.size), "median": float(np.median(a)),
            "q25": float(np.quantile(a, 0.25)), "q75": float(np.quantile(a, 0.75))}


def main():
    split = json.loads((DATA / "split_manifest.json").read_text())
    assigned = {item["path"]: item["partition"] for item in split["files"]}
    index = json.loads((DATA / "dataset_index.json").read_text())
    features = {
        (part, gesture): {name: [] for name in (
            "EMG CH1 RMS", "EMG CH2 RMS", "EMG CH3 RMS",
            "Accel X mean", "Accel Y mean", "Accel Z mean",
            "Accel X SD", "Accel Y SD", "Accel Z SD")}
        for part in PARTS for gesture in CLASSES
    }
    files_used = {part: [] for part in PARTS}
    for gesture in CLASSES:
        for session in index["subjects"]["abinash_s02"][gesture]["sessions"]:
            rel = f"abinash_s02/{gesture}/{session['csv']}"
            part = assigned.get(rel)
            if part not in PARTS or not session.get("include_in_training", False):
                continue
            path = DATA / rel
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            names = ("ch1", "ch2", "ch3", "accel_x", "accel_y", "accel_z")
            data = np.asarray([[float(row[name]) for name in names] for row in rows], dtype=np.float32)
            intervals = load_intervals(path, len(data))
            files_used[part].append(rel)
            for w in iter_windows(data, intervals):
                vals = [float(np.sqrt(np.mean(w[:, i] ** 2))) for i in range(3)]
                vals += [float(np.mean(w[:, i])) for i in range(3, 6)]
                vals += [float(np.std(w[:, i])) for i in range(3, 6)]
                for feature, value in zip(features[(part, gesture)], vals):
                    features[(part, gesture)][feature].append(value)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    fig.suptitle("S02 train vs validation · cue-window feature distributions", fontsize=16, y=0.985)
    fig.text(0.5, 0.955,
             "500-sample windows, 250-sample stride; held-out test recordings are not included",
             ha="center", fontsize=10)
    colors = {"train": "#2463a6", "validation": "#df7b19"}
    feature_order = list(features[("train", "rest")])
    result = {"window_size_samples": WINDOW, "stride_samples": STRIDE,
              "partitions_included": list(PARTS), "test_data_used": False,
              "files": files_used, "features": {}}
    for ax, name in zip(axes.flat, feature_order):
        positions, data_boxes, box_colors, labels = [], [], [], []
        pos = 1
        summary = {}
        for gesture in CLASSES:
            for part in PARTS:
                values = features[(part, gesture)][name]
                if values:
                    positions.append(pos)
                    data_boxes.append(values)
                    box_colors.append(colors[part])
                    labels.append((gesture, part, len(values)))
                    summary[f"{gesture}/{part}"] = summarize(values)
                pos += 1
            pos += 1
        bp = ax.boxplot(data_boxes, positions=positions, widths=0.62,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color("#333333")
            med.set_linewidth(1.2)
        ax.set_xticks([1.5, 4.5, 7.5, 10.5], CLASSES)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        if name.startswith("EMG"):
            ax.set_ylabel("filtered signal units (RMS)")
        elif "mean" in name:
            ax.set_ylabel("signed accelerometer counts/window")
        else:
            ax.set_ylabel("accelerometer counts (SD/window)")
        result["features"][name] = summary
    handles = [plt.Line2D([0], [0], color=colors[p], lw=5, alpha=0.45, label=p.title()) for p in PARTS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0.04, 0.06, 0.99, 0.92))
    png = OUT / "train_validation_feature_shift.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    js = OUT / "train_validation_feature_shift.json"
    js.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"plot": str(png), "summary": str(js),
                      "windows": {part: {gesture: len(features[(part, gesture)][feature_order[0]])
                                         for gesture in CLASSES} for part in PARTS}}, indent=2))


if __name__ == "__main__":
    main()
