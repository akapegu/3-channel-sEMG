"""Fit the selected development candidate on all three development sessions.

This creates a provisional model for a future, untouched Session 4 evaluation.
It does not create a held-out score or imply that project pass criteria are met.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
SESSIONS = ["session1", "session2", "session3"]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for i in (1, 2, 3):
        parser.add_argument(f"--s{i}-csv", type=Path, required=True)
        parser.add_argument(f"--s{i}-cue", type=Path, required=True)
    parser.add_argument("--model-out", type=Path,
                        default=HERE / "model_session123" / "extra_trees_fusion_400ms.joblib")
    parser.add_argument("--metadata-out", type=Path,
                        default=HERE / "model_session123" / "candidate_metadata.json")
    args = parser.parse_args()
    ev = load_evaluator()
    ev.WINDOW, ev.HOP = WINDOW, HOP
    feature_parts, label_parts, metadata = [], [], {}

    for i, session in enumerate(SESSIONS, start=1):
        csv_path = getattr(args, f"s{i}_csv").resolve()
        cue_path = getattr(args, f"s{i}_cue").resolve()
        cue = json.loads(cue_path.read_text())
        frame = pd.read_csv(csv_path)
        if cue.get("status") != "complete" or cue.get("session") != session:
            raise ValueError(f"Expected a complete {session} cue file")
        intervals = ev.get_action_intervals(cue, len(frame))
        counts = collections.Counter((x["trial_type"], x["label"], x["direction"]) for x in intervals)
        expected = collections.Counter({("gesture", c, c): 20 for c in GESTURES})
        expected.update({("tilt_only_control", "rest", c): 10 for c in GESTURES})
        if counts != expected:
            raise ValueError(f"Unexpected cue counts in {session}: {dict(counts)}")
        processing = json.loads(csv_path.with_suffix(".processing.json").read_text())
        if processing.get("sampling_rate") != 1000 or processing.get("preprocessing_profile") != "npg-native1000-bs48_52-hp70-v1":
            raise ValueError(f"Unexpected acquisition profile in {session}")
        x, y, _ = ev.make_windows(frame, intervals, "FUSION")
        feature_parts.append(x)
        label_parts.append(y)
        metadata[session] = {
            "sample_rows": len(frame),
            "window_count": len(y),
            "cue_count": len(intervals),
            "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "cue_sha256": hashlib.sha256(cue_path.read_bytes()).hexdigest(),
        }

    x_train = np.concatenate(feature_parts)
    y_train = np.concatenate(label_parts)
    model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                max_features="sqrt", class_weight="balanced",
                                random_state=2103, n_jobs=-1)
    model.fit(x_train, y_train)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    report = {
        "status": "provisional development candidate; no independent test score",
        "training_sessions": SESSIONS,
        "training_windows": int(len(y_train)),
        "training_windows_by_class": {c: int(np.sum(y_train == c)) for c in model.classes_},
        "model": "ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, max_features=sqrt, class_weight=balanced, random_state=2103)",
        "features": "400-sample (400 ms nominal) windows, 20-sample (20 ms nominal) hop; per-channel log1p RMS, MAV, mean absolute waveform length, plus accelerometer-axis mean/std",
        "fitted_model_path": str(args.model_out.resolve()),
        "session_sources": metadata,
        "limits": [
            "Leave-one-session-out candidate screen was below the 90% event-accuracy target.",
            "Session 4 remains untouched and is required for the final evaluation.",
            "This fit has no held-out accuracy; do not treat training predictions as validation.",
        ],
    }
    args.metadata_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model": str(args.model_out.resolve()),
                      "metadata": str(args.metadata_out.resolve()),
                      "training_windows": len(y_train),
                      "windows_by_class": report["training_windows_by_class"]}, indent=2))


if __name__ == "__main__":
    main()
