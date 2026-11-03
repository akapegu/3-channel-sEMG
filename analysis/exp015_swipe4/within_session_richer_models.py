"""Exploratory richer-model comparison on Session 1, with whole-cue CV groups.

All candidates use the existing per-window handcrafted features. This probes
nonlinear model capacity; it is not a new locked model or re-wear estimate.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
CLASSES = ["rest", "down", "left", "right", "top"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="local continuous processed EMG/IMU CSV")
    parser.add_argument("--cue", type=Path, required=True, help="matching local cue sidecar JSON")
    parser.add_argument("--output", type=Path, default=HERE / "session1_richer_models.json")
    args = parser.parse_args()

    cv_path = HERE / "within_session_grouped_cv.py"
    spec = importlib.util.spec_from_file_location("grouped_cv", cv_path)
    cv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cv)
    ev = cv.load_evaluator()
    frame = pd.read_csv(args.csv)
    cue = json.loads(args.cue.read_text())
    intervals = ev.get_action_intervals(cue, len(frame))
    x_modes, y, ranges = {}, None, {}
    for mode in ("EMG", "FUSION"):
        x_modes[mode], labels, ranges[mode] = ev.make_windows(frame, intervals, mode)
        if y is None:
            y = labels
        elif not np.array_equal(y, labels):
            raise ValueError("Feature modes produced different label order")
    groups = np.concatenate([
        np.full(end - begin, event["trial_index"], dtype=int)
        for event, (begin, end) in zip(intervals, ranges["FUSION"])
    ])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2103)
                 .split(np.zeros((len(y), 1)), y, groups))

    def model_for(name):
        if name == "SVM_RBF":
            return make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf", class_weight="balanced"))
        if name == "RANDOM_FOREST":
            return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                          max_features="sqrt", class_weight="balanced_subsample",
                                          random_state=2103, n_jobs=-1)
        if name == "EXTRA_TREES":
            return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                        max_features="sqrt", class_weight="balanced",
                                        random_state=2103, n_jobs=-1)
        if name == "MLP_32_16":
            return make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=1.0, learning_rate_init=0.001,
                max_iter=1200, early_stopping=False, random_state=2103,
            ))
        raise ValueError(name)

    candidates = [(family, mode) for family in
                  ("SVM_RBF", "RANDOM_FOREST", "EXTRA_TREES", "MLP_32_16")
                  for mode in ("EMG", "FUSION")]
    results = {}
    for family, mode in candidates:
        x = x_modes[mode]
        out = np.empty(len(y), dtype=object)
        fold_event_accuracy = []
        for train, valid in folds:
            model = model_for(family)
            model.fit(x[train], y[train])
            out[valid] = model.predict(x[valid])
            val_groups = set(np.unique(groups[valid]).tolist())
            event_indices = [i for i, event in enumerate(intervals)
                             if event["trial_index"] in val_groups]
            good = total = 0
            for i in event_indices:
                begin, end = ranges[mode][i]
                vote = max(CLASSES, key=lambda label: int(np.sum(out[begin:end] == label)))
                good += vote == intervals[i]["label"]
                total += 1
            fold_event_accuracy.append(good / total)
        event_true, event_pred, _ = cv.event_predictions(out, ranges[mode], intervals)
        summary = cv.summarize(y, out, intervals, ranges[mode])
        results[f"{family}_{mode}"] = {
            "pooled_out_of_fold": summary,
            "fold_event_accuracy": [round(x, 4) for x in fold_event_accuracy],
            "fold_event_accuracy_mean": float(np.mean(fold_event_accuracy)),
            "fold_event_accuracy_std": float(np.std(fold_event_accuracy)),
        }

    report = {
        "evaluation": "5-fold stratified group CV on Session 1 action repetitions",
        "group_unit": "entire prompted action repetition; no overlapping windows cross folds",
        "session_role": "exploratory development only; one wearing; candidate comparison is not an independent test",
        "feature_set": "same 200 ms-window log RMS/MAV/waveform length; fusion adds accelerometer mean/std",
        "model_families": ["RBF SVM", "Random Forest", "Extra Trees", "two-layer MLP (32,16)"],
        "selection_caution": "The same folds are used to screen candidates, so a highest score is optimistic. Do not lock/report it as final performance.",
        "limitations": [
            "Same session and electrode placement appear in all folds.",
            "Cue boundaries are visual prompt times; movement onset is not separately annotated.",
            "The MLP sees correlated windows and no event-balanced sample weighting; its result is a capacity probe only.",
        ],
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {
        "event_accuracy": round(v["pooled_out_of_fold"]["event_accuracy"], 4),
        "gesture_accuracy": round(v["pooled_out_of_fold"]["gesture_event_accuracy"], 4),
        "tilt_false_commands": v["pooled_out_of_fold"]["tilt_only_false_commands"],
        "event_macro_f1": round(v["pooled_out_of_fold"]["event_macro_f1"], 4),
        "fold_accuracy": v["fold_event_accuracy"],
    } for name, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
