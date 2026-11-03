"""Evaluate frozen/previous-session swipe baselines on a new cue-labeled session.

Models fit here use only the original TRAIN+VALIDATION partitions. The new
session is never used to fit scalers or classifiers. This is an exploratory
development-session comparison; it is not the final Session 4 test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
CLASSES = ["rest", "down", "left", "right", "top"]
GESTURES = ["down", "left", "right", "top"]
WINDOW = 200
HOP = 20
CHANNELS = ["ch1", "ch2", "ch3"]
ACCEL = ["accel_x", "accel_y", "accel_z"]


def load_trainer():
    path = HERE / "train_swipe4_baseline.py"
    spec = importlib.util.spec_from_file_location("swipe4_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def accel_features(x):
    a = x[:, 3:6].astype(np.float64)
    return np.r_[np.mean(a, axis=0), np.std(a, axis=0)]


def candidate_model(family, modality):
    if family == "LDA":
        classifier = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=[1 / len(CLASSES)] * len(CLASSES))
    elif family == "SVM":
        classifier = SVC(C=1.0, kernel="rbf", class_weight="balanced")
    else:
        raise ValueError(family)
    return make_pipeline(StandardScaler(), classifier)


def get_action_intervals(cue, n_samples):
    starts = {}
    intervals = []
    for event in cue.get("events", []):
        if event.get("phase") != "ACTION" or event.get("event") not in ("label_start", "label_stop"):
            continue
        key = int(event["phase_index"])
        if event["event"] == "label_start":
            if key in starts:
                raise ValueError(f"Duplicate action start for phase {key}")
            starts[key] = event
            continue
        start = starts.pop(key, None)
        if start is None:
            raise ValueError(f"Action stop without start for phase {key}")
        if (start.get("label") != event.get("label") or
                start.get("trial_index") != event.get("trial_index")):
            raise ValueError(f"Action cue changed metadata for phase {key}")
        a, b = int(start["sample_index"]), int(event["sample_index"])
        if not (0 <= a < b <= n_samples):
            raise ValueError(f"Action interval [{a},{b}) outside {n_samples} samples")
        intervals.append({
            "start": a, "stop": b, "label": start["label"],
            "trial_type": start.get("trial_type"),
            "direction": start.get("direction"),
            "trial_index": start.get("trial_index"),
        })
    if starts:
        raise ValueError(f"Unclosed action cues: {sorted(starts)}")
    intervals.sort(key=lambda x: x["start"])
    return intervals


def make_windows(frame, intervals, use_accel):
    features, labels, event_window_ranges = [], [], []
    columns = CHANNELS + ACCEL
    x = frame[columns].to_numpy(dtype=np.float64)
    for event_no, event in enumerate(intervals):
        begin = len(features)
        for start in range(event["start"], event["stop"] - WINDOW + 1, HOP):
            segment = x[start:start + WINDOW]
            if use_accel == "EMG":
                features.append(TRAINER.window_features(segment, False))
            elif use_accel == "FUSION":
                features.append(TRAINER.window_features(segment, True))
            else:
                features.append(accel_features(segment))
            labels.append(event["label"])
        event_window_ranges.append((begin, len(features)))
    return np.asarray(features), np.asarray(labels), event_window_ranges


def score_model(model, x, y, intervals, event_ranges):
    pred = model.predict(x)
    matrix = confusion_matrix(y, pred, labels=CLASSES)
    window_report = classification_report(
        y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    event_true, event_pred = [], []
    event_rows = []
    for event, (begin, end) in zip(intervals, event_ranges):
        if end <= begin:
            continue
        event_prediction = max(CLASSES, key=lambda label: int(np.sum(pred[begin:end] == label)))
        event_true.append(event["label"])
        event_pred.append(event_prediction)
        event_rows.append({
            "trial_index": event["trial_index"],
            "trial_type": event["trial_type"],
            "direction": event["direction"],
            "true_label": event["label"],
            "predicted_label": event_prediction,
            "correct": event_prediction == event["label"],
        })
    event_matrix = confusion_matrix(event_true, event_pred, labels=CLASSES)
    tilt_rows = [row for row in event_rows if row["trial_type"] == "tilt_only_control"]
    gesture_rows = [row for row in event_rows if row["trial_type"] == "gesture"]
    return {
        "windows_scored": int(len(y)),
        "window_accuracy": float(accuracy_score(y, pred)),
        "window_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "window_macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "window_confusion_labels": CLASSES,
        "window_confusion_true_rows_pred_columns": matrix.tolist(),
        "window_per_class": {c: {m: float(window_report[c][m]) for m in ("precision", "recall", "f1-score", "support")} for c in CLASSES},
        "action_intervals_scored": len(event_rows),
        "event_accuracy": float(accuracy_score(event_true, event_pred)),
        "event_balanced_accuracy": float(balanced_accuracy_score(event_true, event_pred)),
        "event_macro_f1": float(f1_score(event_true, event_pred, labels=CLASSES, average="macro", zero_division=0)),
        "event_confusion_true_rows_pred_columns": event_matrix.tolist(),
        "event_per_class": {
            c: {
                "support": sum(row["true_label"] == c for row in event_rows),
                "correct": sum(row["true_label"] == c and row["correct"] for row in event_rows),
                "recall": (sum(row["true_label"] == c and row["correct"] for row in event_rows) /
                           max(1, sum(row["true_label"] == c for row in event_rows))),
            } for c in CLASSES
        },
        "gesture_only_event_accuracy": float(accuracy_score(
            [x["true_label"] for x in gesture_rows], [x["predicted_label"] for x in gesture_rows])),
        "tilt_only_controls": {
            "count": len(tilt_rows),
            "false_command_count": sum(row["predicted_label"] != "rest" for row in tilt_rows),
            "false_command_rate": (sum(row["predicted_label"] != "rest" for row in tilt_rows) / len(tilt_rows)
                                   if tilt_rows else None),
            "by_direction": {
                d: {"count": sum(r["direction"] == d for r in tilt_rows),
                    "false_commands": sum(r["direction"] == d and r["predicted_label"] != "rest" for r in tilt_rows)}
                for d in GESTURES
            },
        },
        "event_predictions": event_rows,
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    global TRAINER
    TRAINER = load_trainer()

    source = args.csv.resolve()
    cue_path = args.cue.resolve()
    cue = json.loads(cue_path.read_text())
    if cue.get("status") != "complete" or cue.get("acquisition_id") != "SWIPE4_REWEAR_S1C_20261103":
        raise ValueError("Expected the complete, corrected Session 1 cue recording")
    processing = json.loads(source.with_suffix(".processing.json").read_text())
    if processing.get("sampling_rate") != 1000 or processing.get("preprocessing_profile") != "npg-native1000-bs48_52-hp70-v1":
        raise ValueError("Session preprocessing differs from the current baseline")
    frame = pd.read_csv(source)
    if len(frame) != 512620:
        raise ValueError(f"Unexpected source sample count: {len(frame)}")
    intervals = get_action_intervals(cue, len(frame))
    counts = Counter((event["trial_type"], event["label"], event["direction"]) for event in intervals)
    expected = Counter({("gesture", c, c): 20 for c in GESTURES})
    expected.update({("tilt_only_control", "rest", c): 10 for c in GESTURES})
    if counts != expected:
        raise ValueError(f"Action cue counts differ from plan: {dict(counts)}")

    # Load only the original TRAIN and VALIDATION derivatives. Never read TEST.
    xtr_emg, ytr, _, _ = TRAINER.load_partition("train", False)
    xva_emg, yva, _, _ = TRAINER.load_partition("validation", False)
    xtr_fusion, _, _, _ = TRAINER.load_partition("train", True)
    xva_fusion, _, _, _ = TRAINER.load_partition("validation", True)
    y_old = np.concatenate([ytr, yva])
    train_inputs = {
        "EMG": np.concatenate([xtr_emg, xva_emg]),
        "FUSION": np.concatenate([xtr_fusion, xva_fusion]),
        "ACCEL": np.concatenate([xtr_fusion[:, 9:], xva_fusion[:, 9:]]),
    }
    x_new, y_new, event_ranges = {}, None, None
    event_ranges_by_mode = {}
    for mode in ("EMG", "FUSION", "ACCEL"):
        x_new[mode], labels, event_ranges_by_mode[mode] = make_windows(frame, intervals, mode)
        if y_new is None:
            y_new = labels
        elif not np.array_equal(y_new, labels):
            raise ValueError("Feature modes produced different label order")
    event_ranges = event_ranges_by_mode["FUSION"]

    results = {}
    # This is the exact frozen LDA+fusion artifact selected before Session 1.
    locked_path = HERE / "model_v1" / "locked_model.joblib"
    locked_model = joblib.load(locked_path)
    results["LOCKED_LDA_FUSION"] = score_model(
        locked_model, x_new["FUSION"], y_new, intervals, event_ranges)

    for family, mode in (("LDA_EMG", "EMG"), ("SVM_EMG", "EMG"),
                         ("SVM_FUSION", "FUSION"),
                         ("LDA_ACCEL_ONLY", "ACCEL"), ("SVM_ACCEL_ONLY", "ACCEL")):
        model = candidate_model("LDA" if family.startswith("LDA") else "SVM", mode)
        model.fit(train_inputs[mode], y_old)
        results[family] = score_model(
            model, x_new[mode], y_new, intervals, event_ranges_by_mode[mode])

    result = {
        "evaluation": "cross-session development comparison; current session not used for fitting",
        "fit_source": "original EXP-016 TRAIN+VALIDATION only; original TEST partition was not opened",
        "score_source": str(source), "cue_source": str(cue_path),
        "source_sha256": sha256(source), "cue_sha256": sha256(cue_path),
        "sampling_rate_configured_hz": processing["sampling_rate"],
        "window_samples": WINDOW, "hop_samples": HOP,
        "cue_interval_note": "Scores use all windows wholly inside the 1.5 s visual action cue. Event votes are majority class over correlated windows.",
        "limitations": [
            "Session 1 is a development session used for candidate comparison, not the final locked test.",
            "This is cue-aligned offline scoring, not a live command success, false-activation-per-hour, or latency evaluation.",
            "The user confirmed cue compliance; movement onset was not independently measured.",
            "Window metrics are correlated because windows overlap.",
        ],
        "models": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {
        "window_accuracy": round(m["window_accuracy"], 4),
        "window_macro_f1": round(m["window_macro_f1"], 4),
        "event_accuracy": round(m["event_accuracy"], 4),
        "gesture_only_event_accuracy": round(m["gesture_only_event_accuracy"], 4),
        "tilt_false_commands": m["tilt_only_controls"]["false_command_count"],
        "tilt_count": m["tilt_only_controls"]["count"],
    } for name, m in results.items()}, indent=2))


if __name__ == "__main__":
    main()
