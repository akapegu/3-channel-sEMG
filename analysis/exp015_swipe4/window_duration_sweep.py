"""Compare window durations with fixed hop, features, models, and event folds.

This is an exploratory single-wearing study. It compares 100–400 ms windows
using complete action repetitions as outer folds; it does not estimate re-wear
performance or live action latency.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
DURATIONS_MS = [100, 150, 200, 250, 300, 400]
HOP_SAMPLES = 20
CLASSES = ["rest", "down", "left", "right", "top"]


def make_model(family):
    if family == "SVM_RBF":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced"))
    if family == "EXTRA_TREES":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                    max_features="sqrt", class_weight="balanced",
                                    random_state=2103, n_jobs=-1)
    raise ValueError(family)


def build_windows(ev, frame, intervals, mode, window_samples):
    ev.WINDOW = window_samples
    ev.HOP = HOP_SAMPLES
    x, y, event_ranges = ev.make_windows(frame, intervals, mode)
    groups = np.concatenate([
        np.full(end - begin, int(event["trial_index"]), dtype=int)
        for event, (begin, end) in zip(intervals, event_ranges)
    ])
    return x, y, event_ranges, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="local continuous processed EMG/IMU CSV")
    parser.add_argument("--cue", type=Path, required=True, help="matching local cue sidecar JSON")
    parser.add_argument("--output", type=Path, default=HERE / "session1_window_duration_sweep.json")
    args = parser.parse_args()

    cv_path = HERE / "within_session_grouped_cv.py"
    spec = importlib.util.spec_from_file_location("grouped_cv", cv_path)
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    ev = cv.load_evaluator()
    frame = pd.read_csv(args.csv)
    cue = json.loads(args.cue.read_text())
    intervals = ev.get_action_intervals(cue, len(frame))

    event_y = np.asarray([event["label"] for event in intervals])
    event_groups = np.asarray([int(event["trial_index"]) for event in intervals])
    event_folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2103)
                        .split(np.zeros((len(event_y), 1)), event_y, event_groups))

    results = {}
    for duration_ms in DURATIONS_MS:
        n = duration_ms  # acquisition's nominal 1000 samples/s
        for mode in ("EMG", "FUSION"):
            x, y, event_ranges, window_groups = build_windows(ev, frame, intervals, mode, n)
            if not np.array_equal(y, np.concatenate([
                np.full(b - a, event["label"], dtype=object)
                for event, (a, b) in zip(intervals, event_ranges)
            ])):
                raise ValueError("Window labels do not match action intervals")
            windows_by_group = {
                group: np.flatnonzero(window_groups == group)
                for group in np.unique(window_groups)
            }
            for family in ("SVM_RBF", "EXTRA_TREES"):
                oof = np.empty(len(y), dtype=object)
                fold_scores = []
                for train_events, valid_events in event_folds:
                    train_event_ids = {int(event_groups[i]) for i in train_events}
                    valid_event_ids = {int(event_groups[i]) for i in valid_events}
                    train_idx = np.concatenate([windows_by_group[g] for g in train_event_ids])
                    valid_idx = np.concatenate([windows_by_group[g] for g in valid_event_ids])
                    model = make_model(family)
                    model.fit(x[train_idx], y[train_idx])
                    oof[valid_idx] = model.predict(x[valid_idx])
                    good = total = 0
                    for event_no in valid_events:
                        begin, end = event_ranges[event_no]
                        vote = max(CLASSES, key=lambda c: int(np.sum(oof[begin:end] == c)))
                        good += vote == intervals[event_no]["label"]
                        total += 1
                    fold_scores.append(good / total)

                summary = cv.summarize(y, oof, intervals, event_ranges)
                key = f"{family}_{mode}_{duration_ms}ms"
                results[key] = {
                    "duration_ms": duration_ms,
                    "hop_ms": HOP_SAMPLES,
                    "family": family,
                    "modality": mode,
                    "pooled_out_of_fold": summary,
                    "fold_event_accuracy": [round(v, 4) for v in fold_scores],
                    "fold_event_accuracy_mean": float(np.mean(fold_scores)),
                    "fold_event_accuracy_std": float(np.std(fold_scores)),
                }

    report = {
        "evaluation": "5-fold stratified group CV on Session 1 action repetitions",
        "sampling_rate_assumption_hz": 1000,
        "durations_ms": DURATIONS_MS,
        "fixed_hop_ms": HOP_SAMPLES,
        "fixed_features": "per-channel log1p RMS, MAV, mean absolute waveform length; fusion adds accelerometer axis mean/std",
        "fixed_models": "RBF SVM and 400-tree Extra Trees; same hyperparameters at every duration",
        "folds": "fixed event-level StratifiedGroupKFold splits shared by every duration/model/modality",
        "session_role": "development-only; one wearing; candidate results are not an independent test",
        "latency_note": "Event-vote scores pool windows over the full prompted interval and do not measure live onset delay. Live replay latency must be evaluated separately.",
        "limitations": [
            "Same placement and session appear in all folds; selection over candidates is optimistic.",
            "Cue intervals begin at visual cue dispatch; movement onset is not separately annotated.",
            "The 40 REST events are relaxed-finger tilt-only controls, not ordinary-activity exposure.",
        ],
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    compact = {}
    for name, block in results.items():
        score = block["pooled_out_of_fold"]
        compact[name] = {
            "event_accuracy": round(score["event_accuracy"], 4),
            "gesture_accuracy": round(score["gesture_event_accuracy"], 4),
            "tilt_false_commands": score["tilt_only_false_commands"],
            "macro_f1": round(score["event_macro_f1"], 4),
            "fold_accuracy": block["fold_event_accuracy"],
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
