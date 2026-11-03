"""Grouped cross-validation within one four-swipe recording.

Every cue interval is held out as a group, so overlapping windows from one
movement never appear on both sides of a fold. This estimates separability
within this wearing; it does not measure re-wear generalization.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
CLASSES = ["rest", "down", "left", "right", "top"]


def load_evaluator():
    p = HERE / "evaluate_rewear_swipe4.py"
    spec = importlib.util.spec_from_file_location("rewear_eval", p)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.TRAINER = module.load_trainer()
    return module


def event_predictions(pred, event_ranges, intervals):
    truth, guesses, rows = [], [], []
    for event, (begin, end) in zip(intervals, event_ranges):
        vote = max(CLASSES, key=lambda name: int(np.sum(pred[begin:end] == name)))
        truth.append(event["label"])
        guesses.append(vote)
        rows.append({**event, "predicted_label": vote, "correct": vote == event["label"]})
    return truth, guesses, rows


def summarize(y, pred, intervals, event_ranges):
    true_event, pred_event, rows = event_predictions(pred, event_ranges, intervals)
    cm = confusion_matrix(true_event, pred_event, labels=CLASSES)
    per_class = {}
    for i, name in enumerate(CLASSES):
        support = int(cm[i].sum())
        per_class[name] = {"correct": int(cm[i, i]), "support": support,
                           "recall": float(cm[i, i] / support) if support else None}
    controls = [r for r in rows if r.get("trial_type") == "tilt_only_control"]
    gestures = [r for r in rows if r.get("trial_type") == "gesture"]
    return {
        "window_accuracy": float(accuracy_score(y, pred)),
        "window_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "window_macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "event_count": len(rows),
        "event_accuracy": float(accuracy_score(true_event, pred_event)),
        "event_balanced_accuracy": float(balanced_accuracy_score(true_event, pred_event)),
        "event_macro_f1": float(f1_score(true_event, pred_event, labels=CLASSES, average="macro", zero_division=0)),
        "event_confusion_labels": CLASSES,
        "event_confusion_true_rows_pred_columns": cm.tolist(),
        "event_per_class": per_class,
        "gesture_event_accuracy": float(accuracy_score(
            [r["label"] for r in gestures], [r["predicted_label"] for r in gestures])),
        "tilt_only_control_count": len(controls),
        "tilt_only_false_commands": sum(r["predicted_label"] != "rest" for r in controls),
        "tilt_only_false_command_rate": (sum(r["predicted_label"] != "rest" for r in controls) / len(controls)
                                         if controls else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--cue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ev = load_evaluator()
    source, cue_path = args.csv.resolve(), args.cue.resolve()
    cue = json.loads(cue_path.read_text())
    if cue.get("status") != "complete":
        raise ValueError("Only a complete recording can be evaluated")
    frame = pd.read_csv(source)
    intervals = ev.get_action_intervals(cue, len(frame))
    x_by_mode, y, event_ranges_by_mode = {}, None, {}
    for mode in ("EMG", "FUSION", "ACCEL"):
        x_by_mode[mode], labels, event_ranges_by_mode[mode] = ev.make_windows(frame, intervals, mode)
        if y is None:
            y = labels
        elif not np.array_equal(y, labels):
            raise ValueError("Window label order differs between modalities")
    ranges = event_ranges_by_mode["FUSION"]
    groups = np.concatenate([np.full(end - begin, event["trial_index"], dtype=int)
                             for event, (begin, end) in zip(intervals, ranges)])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2103)
    fold_indices = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    results = {}
    for name, family, mode in (
        ("LDA_EMG", "LDA", "EMG"), ("SVM_EMG", "SVM", "EMG"),
        ("LDA_FUSION", "LDA", "FUSION"), ("SVM_FUSION", "SVM", "FUSION"),
        ("LDA_ACCEL_ONLY", "LDA", "ACCEL"), ("SVM_ACCEL_ONLY", "SVM", "ACCEL"),
    ):
        x = x_by_mode[mode]
        event_ranges = event_ranges_by_mode[mode]
        pred_oof = np.empty(len(y), dtype=object)
        fold_event_accuracies = []
        for fold, (train_idx, val_idx) in enumerate(fold_indices):
            model = ev.candidate_model(family, mode)
            model.fit(x[train_idx], y[train_idx])
            pred_oof[val_idx] = model.predict(x[val_idx])
            held_groups = set(int(g) for g in np.unique(groups[val_idx]))
            held_event_indices = [i for i, event in enumerate(intervals)
                                  if event["trial_index"] in held_groups]
            correct, count = 0, 0
            for i in held_event_indices:
                begin, end = event_ranges[i]
                vote = max(CLASSES, key=lambda name: int(np.sum(pred_oof[begin:end] == name)))
                correct += vote == intervals[i]["label"]
                count += 1
            fold_event_accuracies.append(correct / count)
        if any(value is None for value in pred_oof):
            raise RuntimeError(f"Some windows lack out-of-fold predictions for {name}")
        results[name] = {
            "pooled_out_of_fold": summarize(y, pred_oof, intervals, event_ranges),
            "fold_event_accuracy": [round(score, 4) for score in fold_event_accuracies],
            "fold_event_accuracy_mean": float(np.mean(fold_event_accuracies)),
            "fold_event_accuracy_std": float(np.std(fold_event_accuracies)),
        }

    result = {
        "evaluation": "5-fold stratified group CV on Session 1 action intervals",
        "group_unit": "entire prompted action repetition; all overlapping windows stay in one fold",
        "folds": 5,
        "seed": 2103,
        "labels": CLASSES,
        "session_role": "development-only within-session estimate; same wearing appears in each fold",
        "limitations": [
            "Does not estimate performance after re-wearing; Session 4 remains needed for that.",
            "Cue intervals start at visual cue dispatch; movement onset was not independently measured.",
            "The 40 REST examples are tilt-only controls, not hours of ordinary-activity data.",
        ],
        "models": results,
    }
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {
        "event_acc": round(data["pooled_out_of_fold"]["event_accuracy"], 4),
        "event_macro_f1": round(data["pooled_out_of_fold"]["event_macro_f1"], 4),
        "gesture_acc": round(data["pooled_out_of_fold"]["gesture_event_accuracy"], 4),
        "tilt_false": data["pooled_out_of_fold"]["tilt_only_false_commands"],
        "fold_acc": data["fold_event_accuracy"],
    } for name, data in results.items()}, indent=2))


if __name__ == "__main__":
    main()
