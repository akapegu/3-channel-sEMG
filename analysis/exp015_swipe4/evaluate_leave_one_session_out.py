"""Compare fixed candidates with whole-session holdout across Sessions 1–3.

Each fold trains on two complete re-wear sessions and predicts the third.
The candidates are selected on these same folds, so this is development
validation only; Session 4 must remain untouched for a final evaluation.
"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
SESSIONS = ["session1", "session2", "session3"]
GESTURES = ["down", "left", "right", "top"]
CLASSES = ["rest", "down", "left", "right", "top"]
HOP = 20


def load_evaluator():
    path = HERE / "evaluate_rewear_swipe4.py"
    spec = importlib.util.spec_from_file_location("rewear_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TRAINER = module.load_trainer()
    return module


def validate_session(ev, csv_path, cue_path, expected_session):
    cue = json.loads(cue_path.read_text())
    frame = pd.read_csv(csv_path)
    if cue.get("status") != "complete" or cue.get("session") != expected_session:
        raise ValueError(f"Expected complete cue metadata for {expected_session}")
    intervals = ev.get_action_intervals(cue, len(frame))
    counts = collections.Counter((x["trial_type"], x["label"], x["direction"]) for x in intervals)
    expected = collections.Counter({("gesture", c, c): 20 for c in GESTURES})
    expected.update({("tilt_only_control", "rest", c): 10 for c in GESTURES})
    if counts != expected:
        raise ValueError(f"Unexpected cue distribution for {expected_session}: {dict(counts)}")
    processing = json.loads(csv_path.with_suffix(".processing.json").read_text())
    if processing.get("sampling_rate") != 1000 or processing.get("preprocessing_profile") != "npg-native1000-bs48_52-hp70-v1":
        raise ValueError(f"Unexpected sampling/preprocessing profile for {expected_session}")
    return {"frame": frame, "intervals": intervals,
            "csv": csv_path, "cue": cue_path}


def make_model(family):
    if family == "EXTRA_TREES":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                    max_features="sqrt", class_weight="balanced",
                                    random_state=2103, n_jobs=-1)
    if family == "SVM_RBF":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced"))
    raise ValueError(family)


def event_summary(rows):
    truth = [r["true_label"] for r in rows]
    pred = [r["predicted_label"] for r in rows]
    cm = confusion_matrix(truth, pred, labels=CLASSES)
    p, r, f, support = precision_recall_fscore_support(
        truth, pred, labels=CLASSES, zero_division=0)
    gestures = [x for x in rows if x["trial_type"] == "gesture"]
    controls = [x for x in rows if x["trial_type"] == "tilt_only_control"]
    false_commands = sum(x["predicted_label"] != "rest" for x in controls)
    return {
        "event_count": len(rows),
        "event_accuracy": float(accuracy_score(truth, pred)),
        "event_balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "event_macro_f1": float(f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0)),
        "gesture_event_accuracy": float(accuracy_score(
            [x["true_label"] for x in gestures], [x["predicted_label"] for x in gestures])),
        "tilt_control_count": len(controls),
        "tilt_false_commands": false_commands,
        "tilt_false_command_rate": false_commands / len(controls) if controls else None,
        "per_class": {name: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i]), "correct": int(cm[i, i]),
                              "support": int(support[i])}
                       for i, name in enumerate(CLASSES)},
        "confusion_labels": CLASSES,
        "confusion_true_rows_pred_columns": cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for session in ("s1", "s2", "s3"):
        parser.add_argument(f"--{session}-csv", type=Path, required=True)
        parser.add_argument(f"--{session}-cue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ev = load_evaluator()
    inputs = {}
    for i, session in enumerate(SESSIONS, start=1):
        inputs[session] = validate_session(
            ev, getattr(args, f"s{i}_csv"), getattr(args, f"s{i}_cue"), session)

    # Keep this shortlist small and fixed before the held-session predictions.
    candidates = [
        ("EXTRA_TREES", "FUSION", 200),
        ("EXTRA_TREES", "FUSION", 300),
        ("EXTRA_TREES", "FUSION", 400),
        ("EXTRA_TREES", "EMG", 400),
        ("SVM_RBF", "FUSION", 400),
    ]
    results = {}
    for family, modality, duration in candidates:
        ev.WINDOW, ev.HOP = duration, HOP
        fold_results, pooled_rows = {}, []
        for held_out in SESSIONS:
            train_x_parts, train_y_parts = [], []
            for session in SESSIONS:
                item = inputs[session]
                x, y, _ = ev.make_windows(item["frame"], item["intervals"], modality)
                if session == held_out:
                    x_test, y_test = x, y
                else:
                    train_x_parts.append(x)
                    train_y_parts.append(y)
            train_x = np.concatenate(train_x_parts)
            train_y = np.concatenate(train_y_parts)
            model = make_model(family)
            model.fit(train_x, train_y)
            item = inputs[held_out]
            metrics = ev.score_model(model, x_test, y_test, item["intervals"],
                                     ev.make_windows(item["frame"], item["intervals"], modality)[2])
            fold_results[held_out] = {
                "train_sessions": [s for s in SESSIONS if s != held_out],
                "train_window_count": int(len(train_y)),
                "test_window_count": int(len(y_test)),
                "metrics": metrics,
            }
            pooled_rows.extend(metrics["event_predictions"])
        results[f"{family}_{modality}_{duration}ms"] = {
            "folds": fold_results,
            "pooled_out_of_fold": event_summary(pooled_rows),
        }

    report = {
        "evaluation": "leave-one-session-out across Sessions 1–3; each fold trains on two sessions and evaluates on the third",
        "sessions": SESSIONS,
        "window_hop_ms": HOP,
        "fixed_features": "per-channel log1p RMS, MAV, mean absolute waveform length; fusion adds LIS3DH per-axis mean/std",
        "selection_note": "Candidate scores use the same three held-session folds for screening; selected score is optimistic and this is not a final test.",
        "session4_role": "untouched final evaluation after a fixed pipeline is selected and fitted on Sessions 1–3",
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    compact = {}
    for name, block in results.items():
        s = block["pooled_out_of_fold"]
        compact[name] = {
            "event_accuracy": round(s["event_accuracy"], 4),
            "gesture_accuracy": round(s["gesture_event_accuracy"], 4),
            "tilt_false_commands": s["tilt_false_commands"],
            "macro_f1": round(s["event_macro_f1"], 4),
            "per_session_gesture_accuracy": {
                session: round(fold["metrics"]["gesture_only_event_accuracy"], 4)
                for session, fold in block["folds"].items()
            },
            "top_confusion_by_session": {
                session: fold["metrics"]["event_confusion_true_rows_pred_columns"][4]
                for session, fold in block["folds"].items()
            },
        }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
