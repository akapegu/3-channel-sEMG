"""Exploratory read-only comparison of four supplied ChordsWeb swipe CSVs.

The filename is treated as a provisional class label. Each class is only one
recording, so grouped temporal validation describes within-file separability,
not generalization to new repetitions, sessions, or re-wearings.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FILES = ["swipe_down.csv", "swipe_left.csv", "swipe_right.csv", "swipe_top.csv"]
CHANNELS = ["Channel1", "Channel2", "Channel3"]
FS_HZ = 1000  # nominal only; source exports have no timestamps
WINDOW = 200  # samples, 200 ms at nominal rate
HOP = 200      # non-overlapping windows for conservative block validation
BLOCK_S = 2.0


def load_csv(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    if not rows or not {"Counter", *CHANNELS}.issubset(fields):
        raise ValueError(f"Unexpected/empty CSV schema: {path}")
    x = np.asarray([[float(r[c]) for c in CHANNELS] for r in rows], dtype=np.float64)
    counter = np.asarray([int(r["Counter"]) for r in rows], dtype=np.int64)
    return x, counter, fields


def robust_rms(x):
    # Per-channel mean-centred RMS, matching the existing signal review work.
    z = x - np.mean(x, axis=0, keepdims=True)
    return np.sqrt(np.mean(z * z, axis=0))


def get_features(x):
    centered = x - np.mean(x, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=0))
    mav = np.mean(np.abs(centered), axis=0)
    wl = np.mean(np.abs(np.diff(centered, axis=0)), axis=0)
    return np.r_[np.log1p(rms * 1000), np.log1p(mav * 1000), np.log1p(wl * 1000)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    env_rows = []
    feat_rows = []
    features, labels, groups = [], [], []

    for file in FILES:
        path = ROOT / "recordings" / file
        x, counter, columns = load_csv(path)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite EMG values in {path}")
        transitions = (np.diff(counter) % 256)
        label = file.removeprefix("swipe_").removesuffix(".csv")
        records[label] = {
            "source": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": int(len(x)),
            "nominal_duration_s": (len(x) - 1) / FS_HZ,
            "columns": columns,
            "nonunit_counter_transitions": int(np.sum(transitions != 1)),
            "counter_wraps": int(np.sum(np.diff(counter) < 0)),
            "channel_min": {CHANNELS[i]: float(np.min(x[:, i])) for i in range(3)},
            "channel_max": {CHANNELS[i]: float(np.max(x[:, i])) for i in range(3)},
            "channel_global_centered_rms": {CHANNELS[i]: float(robust_rms(x)[i]) for i in range(3)},
        }

        # Plot envelopes every 50 ms, each based on a trailing 200 ms window.
        for end in range(200, len(x) + 1, 50):
            vals = robust_rms(x[end - 200:end])
            t = end / FS_HZ
            for i, channel in enumerate(CHANNELS):
                env_rows.append((label, t, channel, float(vals[i])))

        # Non-overlapping windows keep neighboring windows out of opposite
        # sides of a temporal block boundary in the exploratory CV.
        windows_per_block = int(BLOCK_S * FS_HZ / WINDOW)
        for wi, start in enumerate(range(0, len(x) - WINDOW + 1, HOP)):
            chunk = x[start:start + WINDOW]
            features.append(get_features(chunk))
            labels.append(label)
            block = wi // windows_per_block
            groups.append(f"{label}_block{block}")
            feat_rows.append((label, start / FS_HZ, (start + WINDOW) / FS_HZ, block,
                              *get_features(chunk)))

    # Trace/envelope overview. Raw traces are decimated for display only;
    # the analysis and CSV artifacts use every source row.
    colors = ["#2463a6", "#dc7b23", "#29945b"]
    fig, axes = plt.subplots(len(FILES), 2, figsize=(15, 12), constrained_layout=True)
    for ri, file in enumerate(FILES):
        label = file.removeprefix("swipe_").removesuffix(".csv")
        x, _, _ = load_csv(ROOT / "recordings" / file)
        t = np.arange(len(x)) / FS_HZ
        ax = axes[ri, 0]
        step = max(1, len(x) // 6000)
        for ch in range(3):
            ax.plot(t[::step], x[::step, ch], lw=.55, alpha=.78, color=colors[ch], label=f"CH{ch+1}")
        ax.set_title(f"{label}: processed export, nominal {len(x)/FS_HZ:.1f} s")
        ax.set_ylabel("Browser value")
        ax.grid(alpha=.2)
        if ri == 0:
            ax.legend(ncol=3, fontsize=8)

        ax = axes[ri, 1]
        subset = [row for row in env_rows if row[0] == label]
        for ch, channel in enumerate(CHANNELS):
            rr = [r for r in subset if r[2] == channel]
            ax.plot([r[1] for r in rr], [r[3] for r in rr], lw=1.1,
                    color=colors[ch], label=f"CH{ch+1}")
        ax.set_title("200 ms centered RMS, updated every 50 ms")
        ax.set_ylabel("RMS")
        ax.grid(alpha=.2)
        if ri == 0:
            ax.legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("Time (s, nominal 1000 SPS)")
    axes[-1, 1].set_xlabel("Time (s, nominal 1000 SPS)")
    fig.suptitle("Four supplied swipe recordings — raw traces and per-channel activation", fontsize=15)
    fig.savefig(OUT / "swipe_signal_overview.png", dpi=170)
    plt.close(fig)

    xfeatures = np.asarray(features)
    y = np.asarray(labels)
    g = np.asarray(groups)
    cv = GroupKFold(n_splits=5)
    predictions = {}
    for name, estimator in [
        ("LDA", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ("RBF_SVM", SVC(C=1.0, kernel="rbf", class_weight="balanced")),
    ]:
        model = make_pipeline(StandardScaler(), estimator)
        pred = cross_val_predict(model, xfeatures, y, cv=cv, groups=g)
        predictions[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "confusion_matrix_labels_order": sorted(set(labels)),
            "confusion_matrix": confusion_matrix(y, pred, labels=sorted(set(labels))).tolist(),
        }

    feature_names = [*(f"log_rms_{c}" for c in CHANNELS),
                     *(f"log_mav_{c}" for c in CHANNELS),
                     *(f"log_wl_{c}" for c in CHANNELS)]
    with (OUT / "window_features.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "start_s", "end_s", "temporal_block", *feature_names])
        w.writerows(feat_rows)
    with (OUT / "envelopes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "time_s_nominal", "channel", "mean_centered_rms"])
        w.writerows(env_rows)

    summary = {
        "source_label_assumption": "Each filename is treated as its motion label; no independent cue labels are supplied.",
        "labeling_caveat": "Every non-overlapping window in each file inherited the filename label, including low-activity/rest and transition portions. Annotate actual movement intervals before interpreting scores as swipe classification.",
        "sampling_rate_assumption_hz": FS_HZ,
        "window_ms": WINDOW,
        "hop_ms": HOP,
        "features": feature_names,
        "recordings": records,
        "exploratory_within_recording_group_cv": {
            "grouping": "2-second contiguous blocks within each recording; windows are non-overlapping.",
            "warning": "One recording per class: folds share each source recording across train/test blocks. This is within-take separability only, not independent-repetition, session, re-wear, or user generalization.",
            "models": predictions,
            "window_count": int(len(y)),
            "chance_accuracy": 0.25,
        },
        "artifacts": ["swipe_signal_overview.png", "window_features.csv", "envelopes.csv", "analyse_swipe_classes.py"],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
