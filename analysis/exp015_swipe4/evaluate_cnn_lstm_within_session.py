"""Estimate within-wearing CNN-LSTM accuracy with complete cues held out.

Each of the four recordings is evaluated separately. Five-fold assignment is
stratified by cue label, and all overlapping windows from one cued repetition
stay in the same fold. This is a same-placement diagnostic, not a cross-session
or final test.
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow import keras

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parents[1] / "supplier_toolkit" / "upstream"
sys.path.insert(0, str(UPSTREAM))
CLASSES = ["rest", "down", "left", "right", "top"]
GESTURES = ["down", "left", "right", "top"]
SESSIONS = ["session1", "session2", "session3", "session4"]
FOLDS = 5
WINDOW = 500
HOP = 250
SEED = 2105


def load_evaluator():
    path = HERE / "evaluate_rewear_swipe4.py"
    spec = importlib.util.spec_from_file_location("rewear_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_records(ev, qc_dir, modality):
    records = []
    for session in SESSIONS:
        qc = json.loads((qc_dir / f"{session}_qc.json").read_text())
        csv_path = Path(qc["source_csv"])
        raw_path = Path(qc["raw_packet_file"])
        cue_path = raw_path.with_name(raw_path.stem + ".session_cues.json")
        cue = json.loads(cue_path.read_text())
        if cue.get("status") != "complete" or cue.get("session") != session:
            raise ValueError(f"Invalid cue file for {session}")
        columns = ["ch1", "ch2", "ch3"]
        if modality == "FUSION":
            columns.extend(["accel_x", "accel_y", "accel_z"])
        frame = pd.read_csv(csv_path, usecols=columns)
        intervals = ev.get_action_intervals(cue, len(frame))
        counts = collections.Counter((e["trial_type"], e["label"], e["direction"]) for e in intervals)
        expected = collections.Counter({("gesture", g, g): 20 for g in GESTURES})
        expected.update({("tilt_only_control", "rest", g): 10 for g in GESTURES})
        if counts != expected:
            raise ValueError(f"Unexpected cue counts in {session}: {dict(counts)}")

        x, y, event_ranges = [], [], []
        event_meta = []
        for event_no, event in enumerate(intervals):
            first = len(x)
            for start in range(event["start"], event["stop"] - WINDOW + 1, HOP):
                x.append(frame.iloc[start:start + WINDOW].to_numpy(dtype=np.float32))
                y.append(event["label"])
            last = len(x)
            if first == last:
                raise ValueError(f"Cue {event_no} too short in {session}")
            event_meta.append({
                "event_no": event_no, "trial_index": event["trial_index"],
                "trial_type": event["trial_type"], "direction": event["direction"],
                "label": event["label"], "window_start": first, "window_stop": last,
            })
        records.append({"name": session, "x": np.stack(x), "y": np.asarray(y), "events": event_meta})
    return records


def assign_cue_folds(events, seed):
    rng = np.random.default_rng(seed)
    by_class = collections.defaultdict(list)
    for e in events:
        by_class[e["label"]].append(e["event_no"])
    fold_for_event = {}
    for label, ids in by_class.items():
        ids = list(np.asarray(ids)[rng.permutation(len(ids))])
        for index, event_id in enumerate(ids):
            fold_for_event[int(event_id)] = index % FOLDS
    return fold_for_event


def window_indices(events, event_ids):
    return np.concatenate([
        np.arange(e["window_start"], e["window_stop"])
        for e in events if e["event_no"] in event_ids
    ])


def event_predictions(model, rec, test_indices, mean, std):
    probs = model.predict((rec["x"][test_indices] - mean) / std, batch_size=32, verbose=0)
    pred_labels = np.asarray(CLASSES)[np.argmax(probs, axis=1)]
    local_idx = {int(global_i): local_i for local_i, global_i in enumerate(test_indices)}
    rows = []
    for event in rec["events"]:
        indices = [i for i in range(event["window_start"], event["window_stop"]) if i in local_idx]
        if not indices:
            continue
        labels = pred_labels[[local_idx[i] for i in indices]]
        pred = max(CLASSES, key=lambda c: int(np.sum(labels == c)))
        rows.append({
            "session": rec["name"], "trial_index": event["trial_index"],
            "trial_type": event["trial_type"], "direction": event["direction"],
            "true_label": event["label"], "predicted_label": pred,
            "correct": pred == event["label"],
        })
    return rows


def summarize(rows):
    truth = [r["true_label"] for r in rows]
    pred = [r["predicted_label"] for r in rows]
    matrix = confusion_matrix(truth, pred, labels=CLASSES)
    gestures = [r for r in rows if r["trial_type"] == "gesture"]
    controls = [r for r in rows if r["trial_type"] == "tilt_only_control"]
    return {
        "action_intervals": len(rows),
        "event_accuracy": sum(r["correct"] for r in rows) / max(1, len(rows)),
        "gesture_intervals": len(gestures),
        "gesture_accuracy": sum(r["correct"] for r in gestures) / max(1, len(gestures)),
        "tilt_controls": len(controls),
        "tilt_false_commands": sum(r["predicted_label"] != "rest" for r in controls),
        "tilt_false_command_rate": sum(r["predicted_label"] != "rest" for r in controls) / max(1, len(controls)),
        "per_class": {
            label: {"correct": int(matrix[i, i]), "total": int(matrix[i].sum()),
                    "recall": float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0}
            for i, label in enumerate(CLASSES)
        },
        "confusion_labels": CLASSES,
        "confusion_true_rows_pred_columns": matrix.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qc-dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path, default=HERE / "cnn_lstm_sessions" / "within_session_grouped_cv.json")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--modality", choices=("EMG", "FUSION"), default="EMG")
    args = ap.parse_args()

    from model_architectures import build_model

    ev = load_evaluator()
    records = load_records(ev, args.qc_dir.resolve(), args.modality)
    report = {
        "evaluation": "five-fold within-session grouped cue CV; each session is evaluated separately using only its own wearing",
        "model": f"supplier CNN-LSTM; {args.modality} ({3 if args.modality == 'EMG' else 6} channels)",
        "window_samples": WINDOW, "window_ms_nominal": WINDOW,
        "hop_samples": HOP, "hop_ms_nominal": HOP,
        "folds": FOLDS,
        "grouping": "all overlapping windows from one prompted action interval stay in one fold; internal validation also holds out complete cues",
        "sessions": {},
        "limits": [
            "These are same-wearing diagnostic scores, not performance after re-wearing or a locked final test.",
            "Session 4 has already been examined and is development data.",
            "Cued tilt-only controls do not estimate ordinary-activity false-command rate.",
            "Model selection and protocol development used these recordings; scores are exploratory.",
        ],
    }

    for session_i, rec in enumerate(records):
        fold_assignment = assign_cue_folds(rec["events"], SEED + session_i)
        oof_rows = []
        fold_reports = {}
        for fold in range(FOLDS):
            test_event_ids = {e_id for e_id, assigned in fold_assignment.items() if assigned == fold}
            train_event_ids = set(fold_assignment) - test_event_ids
            # Hold out 20% of the available training repetitions per class
            # for early stopping; complete cue groups remain intact.
            rng = np.random.default_rng(SEED + 100 * session_i + fold)
            val_event_ids = set()
            for label in CLASSES:
                candidates = [e["event_no"] for e in rec["events"]
                              if e["event_no"] in train_event_ids and e["label"] == label]
                order = rng.permutation(len(candidates))
                n_val = max(1, round(0.2 * len(candidates)))
                val_event_ids.update(candidates[int(i)] for i in order[:n_val])
            fit_event_ids = train_event_ids - val_event_ids

            fit_idx = window_indices(rec["events"], fit_event_ids)
            val_idx = window_indices(rec["events"], val_event_ids)
            test_idx = window_indices(rec["events"], test_event_ids)
            x_fit, x_val, x_test = rec["x"][fit_idx], rec["x"][val_idx], rec["x"][test_idx]
            mean = x_fit.mean(axis=(0, 1), keepdims=True)
            std = x_fit.std(axis=(0, 1), keepdims=True)
            std[std == 0] = 1.0
            x_fit, x_val = (x_fit - mean) / std, (x_val - mean) / std
            keras.utils.set_random_seed(SEED + 10 * session_i + fold)
            model = build_model("cnn_lstm", WINDOW, rec["x"].shape[2], len(CLASSES))
            y_fit = keras.utils.to_categorical([CLASSES.index(x) for x in rec["y"][fit_idx]], len(CLASSES))
            y_val = keras.utils.to_categorical([CLASSES.index(x) for x in rec["y"][val_idx]], len(CLASSES))
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
            ]
            history = model.fit(x_fit, y_fit, validation_data=(x_val, y_val),
                                epochs=args.epochs, batch_size=32,
                                callbacks=callbacks, verbose=0, shuffle=True)
            rows = event_predictions(model, rec, test_idx, mean, std)
            oof_rows.extend(rows)
            fold_reports[f"fold_{fold + 1}"] = {
                "train_cues": len(fit_event_ids), "validation_cues": len(val_event_ids),
                "test_cues": len(test_event_ids), "epochs_run": len(history.history["loss"]),
                "metrics": summarize(rows),
            }
            print(f"{rec['name']} fold {fold + 1}/{FOLDS}: "
                  f"{summarize(rows)['gesture_accuracy']:.3f} gesture accuracy; "
                  f"{summarize(rows)['tilt_false_commands']}/8 tilt false commands", flush=True)
        report["sessions"][rec["name"]] = {
            "out_of_fold": summarize(oof_rows),
            "folds": fold_reports,
            "event_predictions": oof_rows,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({s: {
        "gesture_accuracy": round(v["out_of_fold"]["gesture_accuracy"], 4),
        "event_accuracy": round(v["out_of_fold"]["event_accuracy"], 4),
        "tilt_false_commands": f"{v['out_of_fold']['tilt_false_commands']}/{v['out_of_fold']['tilt_controls']}",
        "per_class_recall": {c: round(m["recall"], 3) for c, m in v["out_of_fold"]["per_class"].items()},
    } for s, v in report["sessions"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
