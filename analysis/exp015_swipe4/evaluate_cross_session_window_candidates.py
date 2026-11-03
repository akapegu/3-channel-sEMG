"""Train window/model candidates on one wearing and score a separate re-wear.

Session 2 is a development evaluation, not the reserved final test. The script
never fits on the test CSV/cues and requires explicit local file paths.
"""
import argparse
import collections
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
CLASSES = ["rest", "down", "left", "right", "top"]
GESTURES = ["down", "left", "right", "top"]
HOP = 20


def load_evaluator():
    path = HERE / "evaluate_rewear_swipe4.py"
    spec = importlib.util.spec_from_file_location("rewear_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TRAINER = module.load_trainer()
    return module


def validate_cues(ev, cue, row_count, expected_session):
    if cue.get("status") != "complete" or cue.get("session") != expected_session:
        raise ValueError(f"Expected a complete {expected_session} capture")
    intervals = ev.get_action_intervals(cue, row_count)
    counts = collections.Counter((x["trial_type"], x["label"], x["direction"]) for x in intervals)
    expected = collections.Counter({("gesture", c, c): 20 for c in GESTURES})
    expected.update({("tilt_only_control", "rest", c): 10 for c in GESTURES})
    if counts != expected:
        raise ValueError(f"Unexpected cue counts for {expected_session}: {dict(counts)}")
    return intervals


def build_features(ev, frame, intervals, duration_ms, mode):
    ev.WINDOW = int(duration_ms)
    ev.HOP = HOP
    return ev.make_windows(frame, intervals, mode)


def make_model(family):
    if family == "EXTRA_TREES":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                    max_features="sqrt", class_weight="balanced",
                                    random_state=2103, n_jobs=-1)
    if family == "SVM_RBF":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced"))
    raise ValueError(family)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--train-cue", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--test-cue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ev = load_evaluator()
    train_cue = json.loads(args.train_cue.read_text())
    test_cue = json.loads(args.test_cue.read_text())
    train_frame = pd.read_csv(args.train_csv)
    test_frame = pd.read_csv(args.test_csv)
    train_intervals = validate_cues(ev, train_cue, len(train_frame), "session1")
    test_intervals = validate_cues(ev, test_cue, len(test_frame), "session2")

    train_processing = json.loads(args.train_csv.with_suffix(".processing.json").read_text())
    test_processing = json.loads(args.test_csv.with_suffix(".processing.json").read_text())
    if train_processing.get("sampling_rate") != 1000 or test_processing.get("sampling_rate") != 1000:
        raise ValueError("Both captures must have the configured 1000 Hz sample rate")
    if train_processing.get("preprocessing_profile") != test_processing.get("preprocessing_profile"):
        raise ValueError("Training and re-wear preprocessing profiles differ")

    # Each candidate is fully trained on Session 1. Session 2 features are
    # passed only to predict(), never to fit() or scaler fitting.
    candidates = [
        ("EXTRA_TREES", "FUSION", 300),
        ("EXTRA_TREES", "FUSION", 400),
        ("EXTRA_TREES", "EMG", 400),
        ("SVM_RBF", "FUSION", 300),
        ("SVM_RBF", "FUSION", 400),
    ]
    results = {}
    for family, mode, duration in candidates:
        x_train, y_train, _ = build_features(ev, train_frame, train_intervals, duration, mode)
        x_test, y_test, event_ranges = build_features(ev, test_frame, test_intervals, duration, mode)
        model = make_model(family)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        summary = ev.score_model(model, x_test, y_test, test_intervals, event_ranges)
        # score_model predicts internally; assert that this output is identical.
        if not np.array_equal(pred, model.predict(x_test)):
            raise RuntimeError("Prediction output changed unexpectedly")
        results[f"{family}_{mode}_{duration}ms"] = {
            "train_windows": int(len(y_train)),
            "test_windows": int(len(y_test)),
            "test_metrics": summary,
        }

    report = {
        "evaluation": "train on complete Session 1 wearing; score on complete Session 2 re-wearing",
        "training_data": str(args.train_csv.resolve()),
        "test_data": str(args.test_csv.resolve()),
        "training_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "window_hop_ms": HOP,
        "features": "per-channel log1p RMS, MAV, mean absolute waveform length; fusion adds LIS3DH mean and standard deviation",
        "session2_role": "development-only comparison; do not present as final test",
        "limitations": [
            "Only one training wearing and one re-wearing are available.",
            "Window and model candidates are screened on Session 2, so these results are for development; keep Session 4 untouched.",
            "Event votes use the full visual cue interval and do not measure movement-onset latency.",
            "The REST examples are the 40 relaxed-finger tilt-only controls, not ordinary-activity exposure.",
        ],
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: {
        "event_accuracy": round(x["test_metrics"]["event_accuracy"], 4),
        "gesture_accuracy": round(x["test_metrics"]["gesture_only_event_accuracy"], 4),
        "tilt_false_commands": x["test_metrics"]["tilt_only_controls"]["false_command_count"],
        "event_macro_f1": round(x["test_metrics"]["event_macro_f1"], 4),
    } for name, x in results.items()}, indent=2))


if __name__ == "__main__":
    main()
