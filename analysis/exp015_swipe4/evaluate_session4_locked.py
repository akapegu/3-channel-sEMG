"""Score the frozen Session 1–3 candidate once on untouched Session 4."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model_session123" / "extra_trees_fusion_400ms.joblib"
METADATA = HERE / "model_session123" / "candidate_metadata.json"
GESTURES = ["down", "left", "right", "top"]
EXPECTED_CUE_ID = "SWIPE4_REWEAR_S4_FINAL"


def load_evaluator():
    path = HERE / "evaluate_rewear_swipe4.py"
    spec = importlib.util.spec_from_file_location("rewear_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TRAINER = module.load_trainer()
    return module


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--metadata", type=Path, default=METADATA)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    if metadata.get("training_sessions") != ["session1", "session2", "session3"]:
        raise ValueError("Frozen model was not fit on Sessions 1–3")
    if "no independent test score" not in metadata.get("status", ""):
        raise ValueError("Expected the saved provisional all-session candidate metadata")
    cue = json.loads(args.cue.read_text())
    if cue.get("status") != "complete" or cue.get("session") != "session4" or cue.get("acquisition_id") != EXPECTED_CUE_ID:
        raise ValueError("Expected the complete, locked Session 4 capture")
    frame = pd.read_csv(args.csv)
    processing = json.loads(args.csv.with_suffix(".processing.json").read_text())
    if processing.get("sampling_rate") != 1000 or processing.get("preprocessing_profile") != "npg-native1000-bs48_52-hp70-v1":
        raise ValueError("Session 4 acquisition profile differs from training")

    ev = load_evaluator()
    intervals = ev.get_action_intervals(cue, len(frame))
    counts = collections.Counter((x["trial_type"], x["label"], x["direction"]) for x in intervals)
    expected = collections.Counter({("gesture", c, c): 20 for c in GESTURES})
    expected.update({("tilt_only_control", "rest", c): 10 for c in GESTURES})
    if counts != expected:
        raise ValueError(f"Session 4 cue counts differ from the locked protocol: {dict(counts)}")

    ev.WINDOW, ev.HOP = 400, 20
    x_test, y_test, event_ranges = ev.make_windows(frame, intervals, "FUSION")
    model = joblib.load(args.model)
    metrics = ev.score_model(model, x_test, y_test, intervals, event_ranges)
    report = {
        "evaluation": "locked final test; model trained on Sessions 1–3 and scored once on Session 4",
        "model_training_sessions": metadata["training_sessions"],
        "model_sha256": sha256(args.model),
        "session4_csv": str(args.csv.resolve()),
        "session4_csv_sha256": sha256(args.csv),
        "session4_cue_sha256": sha256(args.cue),
        "session4_rows": len(frame),
        "session4_windows": len(y_test),
        "no_fitting_or_tuning_on_session4": True,
        "metrics": metrics,
        "limits": [
            "One participant and one final re-wear session; results do not establish generalization to other people or long-term daily use.",
            "Event votes use the entire 1.5 second visual cue and do not measure movement-onset latency.",
            "REST examples are the 40 relaxed-finger tilt controls, not ordinary-activity exposure.",
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
