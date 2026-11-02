"""Train/validate a small time-domain swipe baseline without window leakage.

The validate stage reads only TRAIN and VALIDATION derivatives. The lock
stage refits the validation-selected model on TRAIN+VALIDATION. Only the
separate test stage opens TEST files, after a frozen model has been written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
DERIVED = HERE.parents[1] / "supplier_toolkit" / "adapted" / "training_data" / "abinash_swipe4" / "derived_v2"
OUT = HERE / "model_v1"
CLASSES = ["rest", "down", "left", "right", "top"]
GESTURES = ["down", "left", "right", "top"]
CHANNELS = ["ch1", "ch2", "ch3"]
ACCEL = ["accel_x", "accel_y", "accel_z"]
WINDOW = 200
HOP = 20


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def window_features(x, use_accel):
    emg = x[:, :3].astype(np.float64)
    centered = emg - np.mean(emg, axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centered ** 2, axis=0))
    mav = np.mean(np.abs(centered), axis=0)
    wl = np.mean(np.abs(np.diff(centered, axis=0)), axis=0)
    feat = np.r_[np.log1p(rms), np.log1p(mav), np.log1p(wl)]
    if use_accel:
        accel = x[:, 3:6].astype(np.float64)
        feat = np.r_[feat, np.mean(accel, axis=0), np.std(accel, axis=0)]
    return feat


def load_partition(partition, use_accel):
    features, labels, segment_ids, hashes = [], [], [], {}
    for label in CLASSES:
        csv_path = DERIVED / partition / f"{label}.csv"
        meta_path = csv_path.with_suffix(".segments.json")
        if not csv_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing cue-derived data for {partition}/{label}")
        meta = json.loads(meta_path.read_text())
        if meta.get("label") != label or meta.get("partition") != partition:
            raise ValueError(f"Sidecar label/partition mismatch for {csv_path}")
        if not meta.get("user_cue_compliance_confirmed"):
            raise ValueError(f"Cue compliance not confirmed for {csv_path}")
        df = pd.read_csv(csv_path)
        columns = CHANNELS + ACCEL
        if not set(columns).issubset(df.columns):
            raise ValueError(f"Expected EMG and accelerometer columns in {csv_path}")
        x = df[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values in {csv_path}")
        local_ranges = meta.get("derived_segment_ranges", [])
        if len(local_ranges) != meta.get("segment_count"):
            raise ValueError(f"Segment boundary count mismatch in {meta_path}")
        if len(x) != meta.get("rows"):
            raise ValueError(f"Row count mismatch in {csv_path}")
        for segment_no, (a, b) in enumerate(local_ranges):
            if not (0 <= a < b <= len(x)):
                raise ValueError(f"Bad local range [{a},{b}) in {meta_path}")
            for start in range(a, b - WINDOW + 1, HOP):
                features.append(window_features(x[start:start + WINDOW], use_accel))
                labels.append(label)
                segment_ids.append(f"{label}:{segment_no}")
        hashes[str(csv_path)] = sha256(csv_path)
    return np.asarray(features), np.asarray(labels), np.asarray(segment_ids), hashes


def make_model(name):
    # Candidate names are LDA_EMG, SVM_EMG, LDA_FUSION, and SVM_FUSION.
    estimator, modality = name.split("_", 1)
    if estimator == "LDA":
        classifier = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=[1 / len(CLASSES)] * len(CLASSES))
    elif estimator == "SVM":
        classifier = SVC(C=1.0, kernel="rbf", class_weight="balanced")
    else:
        raise ValueError(name)
    return make_pipeline(StandardScaler(), classifier), modality != "EMG"


def score(model, x, y, segment_ids):
    pred = model.predict(x)
    matrix = confusion_matrix(y, pred, labels=CLASSES)
    report = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    # Each cue interval receives one event-level class by majority vote over
    # the windows wholly inside it. These intervals are the action units;
    # windows within them are correlated, not independent repetitions.
    cue_truth, cue_pred = [], []
    for seg in dict.fromkeys(segment_ids.tolist()):
        indices = np.flatnonzero(segment_ids == seg)
        if not len(indices):
            continue
        truth = y[indices[0]]
        vote = max(CLASSES, key=lambda c: int(np.sum(pred[indices] == c)))
        if truth != "rest":
            cue_truth.append(truth)
            cue_pred.append(vote)
    return {
        "window_count": int(len(y)), "window_accuracy": float(accuracy_score(y, pred)),
        "window_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "window_macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "confusion_matrix_labels": CLASSES, "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
        "per_class": {label: {metric: float(report[label][metric]) for metric in ("precision", "recall", "f1-score", "support")} for label in CLASSES},
        "active_cue_count": len(cue_truth),
        "active_cue_majority_accuracy": float(accuracy_score(cue_truth, cue_pred)) if cue_truth else None,
        "active_cue_confusion_rows_true_columns_pred": confusion_matrix(cue_truth, cue_pred, labels=CLASSES).tolist() if cue_truth else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("validate", "lock", "test"), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    selection_path = OUT / "validation_selection.json"
    model_path = OUT / "locked_model.joblib"

    if args.stage == "validate":
        x_train_emg, y_train, g_train, train_hashes = load_partition("train", False)
        x_train_fusion, _, _, _ = load_partition("train", True)
        x_val_emg, y_val, g_val, val_hashes = load_partition("validation", False)
        x_val_fusion, _, _, _ = load_partition("validation", True)
        candidates = {}
        for name, x_train, x_val in (
            ("LDA_EMG", x_train_emg, x_val_emg),
            ("SVM_EMG", x_train_emg, x_val_emg),
            ("LDA_FUSION", x_train_fusion, x_val_fusion),
            ("SVM_FUSION", x_train_fusion, x_val_fusion),
        ):
            model, _ = make_model(name)
            model.fit(x_train, y_train)
            candidates[name] = score(model, x_val, y_val, g_val)
        selected = max(candidates, key=lambda k: (candidates[k]["window_macro_f1"], candidates[k]["active_cue_majority_accuracy"] or 0))
        result = {
            "stage": "validation_only",
            "feature_config": {"window_samples": WINDOW, "hop_samples": HOP,
                               "nominal_window_ms": WINDOW, "nominal_update_ms": HOP,
                               "emg_features": "per-channel log1p RMS, MAV, mean absolute waveform length",
                               "fusion_features": "the same EMG features plus per-axis accelerometer mean and standard deviation",
                               "feature_scaling": "StandardScaler fitted by each pipeline on TRAIN only"},
            "train_windows": int(len(y_train)), "train_windows_by_class": {c: int(np.sum(y_train == c)) for c in CLASSES},
            "validation_windows": int(len(y_val)), "validation_windows_by_class": {c: int(np.sum(y_val == c)) for c in CLASSES},
            "prompted_repetitions_per_class": {c: 8 for c in GESTURE_CLASSES},
            "candidates": candidates,
            "selected_by_validation_macro_f1": selected,
            "test_loaded": False,
            "warning": "Overlapping windows are correlated; repetition-level majority vote is also reported. All blocks are from one wearing/session.",
            "train_file_sha256": train_hashes, "validation_file_sha256": val_hashes,
            "sources": {"feature_baseline": "Hudgins et al. (1993), https://pubmed.ncbi.nlm.nih.gov/8468080/",
                        "splitting": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection/GroupKFold.html",
                        "project_review": "docs/semg_literature_review.md"},
        }
        selection_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    if not selection_path.exists():
        raise SystemExit("Run --stage validate first; no validation-selected model exists.")
    selection = json.loads(selection_path.read_text())
    if args.stage == "lock":
        selected = selection["selected_by_validation_macro_f1"]
        use_accel = "FUSION" in selected
        x_train, y_train, _, train_hashes = load_partition("train", use_accel)
        x_val, y_val, _, val_hashes = load_partition("validation", use_accel)
        x = np.concatenate([x_train, x_val])
        y = np.concatenate([y_train, y_val])
        model, _ = make_model(selected)
        model.fit(x, y)
        joblib.dump(model, model_path)
        locked = {
            "selected_candidate": selected, "locked_model": str(model_path),
            "feature_config": {**selection["feature_config"],
                               "feature_scaling": "StandardScaler fitted on TRAIN+VALIDATION after validation-selected model choice"},
            "training_partitions": ["train", "validation"],
            "test_read": False, "class_order": CLASSES,
            "train_file_sha256": train_hashes, "validation_file_sha256": val_hashes,
            "selection_source": str(selection_path),
            "lock_note": "Candidate chosen by validation macro-F1; refit once on TRAIN+VALIDATION. TEST remains unopened until --stage test.",
        }
        (OUT / "locked_model.json").write_text(json.dumps(locked, indent=2) + "\n")
        print(json.dumps(locked, indent=2))
        return

    locked_path = OUT / "locked_model.json"
    if not locked_path.exists() or not model_path.exists():
        raise SystemExit("Run --stage lock first; no frozen model exists.")
    locked = json.loads(locked_path.read_text())
    x_test, y_test, g_test, test_hashes = load_partition("test", "FUSION" in locked["selected_candidate"])
    model = joblib.load(model_path)
    metrics = score(model, x_test, y_test, g_test)
    result = {
        "stage": "locked_test_once", "selected_candidate": locked["selected_candidate"],
        "test_windows": int(len(y_test)), "test_windows_by_class": {c: int(np.sum(y_test == c)) for c in CLASSES},
        "metrics": metrics, "test_file_sha256": test_hashes,
        "test_limitations": [
            "This test block is one chronological partition from the same wearing/session as training and validation.",
            "There are only eight prompted repetitions per gesture; window metrics must not be interpreted as independent-trial accuracy.",
            "The cue interval includes visual dispatch/human reaction time and may include post-movement rest.",
            "This is offline window/repetition scoring, not a real-time HID or false-command-rate test.",
        ],
    }
    (OUT / "locked_test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


GESTURE_CLASSES = list(GESTURES)

if __name__ == "__main__":
    main()
