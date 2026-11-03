"""Confirm the selected feature/model settings on a third re-wear session.

Train the fixed 400 ms fused Extra Trees configuration on Sessions 1 and 2,
then score Session 3 once. Session 3 is still development/confirmation data;
Session 4 remains the final untouched evaluation.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
GESTURES = ["down", "left", "right", "top"]
WINDOW = 400
HOP = 20


def load_evaluator():
    path = HERE / "evaluate_rewear_swipe4.py"
    spec = importlib.util.spec_from_file_location("rewear_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TRAINER = module.load_trainer()
    return module


def read_session(ev, csv_path, cue_path, session_name):
    cue = json.loads(cue_path.read_text())
    frame = pd.read_csv(csv_path)
    if cue.get("status") != "complete" or cue.get("session") != session_name:
        raise ValueError(f"Expected complete {session_name} cue file")
    intervals = ev.get_action_intervals(cue, len(frame))
    counts = collections.Counter((x["trial_type"], x["label"], x["direction"]) for x in intervals)
    expected = collections.Counter({("gesture", c, c): 20 for c in GESTURES})
    expected.update({("tilt_only_control", "rest", c): 10 for c in GESTURES})
    if counts != expected:
        raise ValueError(f"Unexpected cue distribution in {session_name}: {dict(counts)}")
    processing = json.loads(csv_path.with_suffix(".processing.json").read_text())
    if processing.get("sampling_rate") != 1000 or processing.get("preprocessing_profile") != "npg-native1000-bs48_52-hp70-v1":
        raise ValueError(f"Unexpected acquisition profile in {session_name}")
    return frame, intervals


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for session in ("s1", "s2", "s3"):
        parser.add_argument(f"--{session}-csv", type=Path, required=True)
        parser.add_argument(f"--{session}-cue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ev = load_evaluator()
    session_inputs = {}
    for name in ("s1", "s2", "s3"):
        session_inputs[name] = read_session(
            ev, getattr(args, f"{name}_csv"), getattr(args, f"{name}_cue"), f"session{name[1]}"
        )

    ev.WINDOW, ev.HOP = WINDOW, HOP
    train_x, train_y = [], []
    for name in ("s1", "s2"):
        frame, intervals = session_inputs[name]
        x, y, _ = ev.make_windows(frame, intervals, "FUSION")
        train_x.append(x)
        train_y.append(y)
    test_frame, test_intervals = session_inputs["s3"]
    test_x, test_y, test_event_ranges = ev.make_windows(test_frame, test_intervals, "FUSION")
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)

    model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=2103, n_jobs=-1)
    model.fit(train_x, train_y)
    metrics = ev.score_model(model, test_x, test_y, test_intervals, test_event_ranges)

    report = {
        "evaluation": "fit on Sessions 1+2; score fixed candidate once on Session 3",
        "selected_candidate": "Extra Trees, EMG+accelerometer, 400 ms feature window, 20 ms hop",
        "training_sessions": ["session1", "session2"],
        "evaluation_session": "session3",
        "training_windows": int(len(train_y)),
        "test_windows": int(len(test_y)),
        "session_rows": {name: int(len(item[0])) for name, item in session_inputs.items()},
        "session_csv_sha256": {
            name: sha256(getattr(args, f"{name}_csv")) for name in ("s1", "s2", "s3")
        },
        "window_ms": WINDOW,
        "hop_ms": HOP,
        "test_metrics": metrics,
        "interpretation_limits": [
            "Session 3 is development/confirmation data; Session 4 remains the final untouched test.",
            "This is one user's three re-wear sessions on nearby dates, not a population or long-term result.",
            "Event votes use all windows in each visual cue and do not measure movement-onset latency.",
            "The 40 REST events are relaxed-finger tilt controls, not ordinary-activity exposure.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "event_accuracy": round(metrics["event_accuracy"], 4),
        "gesture_accuracy": round(metrics["gesture_only_event_accuracy"], 4),
        "tilt_false_commands": metrics["tilt_only_controls"]["false_command_count"],
        "per_class": metrics["event_per_class"],
        "confusion": metrics["event_confusion_true_rows_pred_columns"],
    }, indent=2))


if __name__ == "__main__":
    main()
