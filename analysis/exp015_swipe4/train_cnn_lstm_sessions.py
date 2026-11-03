"""Session-wise CNN-LSTM screen and all-session personalized fit.

Uses four whole re-wear sessions. Each fold leaves one complete session out;
validation cues are grouped by full repetition within the remaining sessions.
No overlapping windows from one cue cross the train/validation/test boundary.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tensorflow import keras

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parents[1] / "supplier_toolkit" / "upstream"
sys.path.insert(0, str(UPSTREAM))
CLASSES = ["rest", "down", "left", "right", "top"]
GESTURES = ["down", "left", "right", "top"]
SESSIONS = ["session1", "session2", "session3", "session4"]
WINDOW = 500  # 500 ms at this project's 1000 Hz processed sample rate
HOP = 250     # supplier default's nominal 50% overlap (250 ms)
SEED = 2104


def load_evaluator():
    path = HERE / "evaluate_rewear_swipe4.py"
    spec = importlib.util.spec_from_file_location("rewear_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_session(ev, name, csv_path: Path, cue_path: Path):
    cue = json.loads(cue_path.read_text())
    if cue.get("status") != "complete" or cue.get("session") != name:
        raise ValueError(f"Expected complete {name} cue sidecar: {cue_path}")
    if cue.get("acquisition_id") not in {
        "SWIPE4_REWEAR_S1C_20261103", "SWIPE4_REWEAR_S2",
        "SWIPE4_REWEAR_S3", "SWIPE4_REWEAR_S4_FINAL",
    }:
        raise ValueError(f"Unexpected acquisition_id in {cue_path}")
    frame = pd.read_csv(csv_path)
    processing = json.loads(csv_path.with_suffix(".processing.json").read_text())
    if processing.get("sampling_rate") != 1000 or processing.get("preprocessing_profile") != "npg-native1000-bs48_52-hp70-v1":
        raise ValueError(f"Unexpected preprocessing profile: {csv_path}")
    intervals = ev.get_action_intervals(cue, len(frame))
    counts = collections.Counter((e["trial_type"], e["label"], e["direction"]) for e in intervals)
    expected = collections.Counter({("gesture", g, g): 20 for g in GESTURES})
    expected.update({("tilt_only_control", "rest", g): 10 for g in GESTURES})
    if counts != expected:
        raise ValueError(f"Unexpected cue counts for {name}: {dict(counts)}")

    columns = ["ch1", "ch2", "ch3"]
    x_source = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(x_source).all():
        raise ValueError(f"Non-finite EMG values in {csv_path}")
    # Keep cue trials intact. The event key maps every overlapping window to
    # exactly one prompted repetition.
    windows, labels, event_ids, event_meta = [], [], [], []
    for event_no, event in enumerate(intervals):
        first = len(windows)
        for start in range(event["start"], event["stop"] - WINDOW + 1, HOP):
            windows.append(x_source[start:start + WINDOW])
            labels.append(event["label"])
            event_ids.append(event_no)
        last = len(windows)
        if first == last:
            raise ValueError(f"Cue {event_no} is too short for one {WINDOW}-sample window")
        event_meta.append({
            "event_no": event_no,
            "trial_index": event["trial_index"],
            "trial_type": event["trial_type"],
            "direction": event["direction"],
            "label": event["label"],
            "window_start": first,
            "window_stop": last,
        })
    return {
        "name": name,
        "x_emg": np.stack(windows),
        "y": np.asarray(labels),
        "event_ids": np.asarray(event_ids),
        "events": event_meta,
        "csv": str(csv_path.resolve()),
        "cue": str(cue_path.resolve()),
        "csv_sha256": sha256(csv_path),
        "cue_sha256": sha256(cue_path),
    }


def make_validation_split(records, fold_seed):
    """Hold out 20% of complete training cues, balanced by class."""
    rng = np.random.default_rng(fold_seed)
    # records is a list of (session_index, record) tuples.
    cue_keys = collections.defaultdict(list)
    for rec_i, rec in records:
        for event in rec["events"]:
            cue_keys[event["label"]].append((rec_i, event))
    selected = set()
    for label, items in cue_keys.items():
        if len(items) < 5:
            raise ValueError(f"Too few whole training cues for validation class {label}")
        order = rng.permutation(len(items))
        for k in order[:max(1, round(len(items) * 0.2))]:
            rec_i, event = items[int(k)]
            selected.update((rec_i, int(i)) for i in range(event["window_start"], event["window_stop"]))

    # Return boolean masks, one per (session_index, session record) tuple.
    masks = {}
    for rec_i, rec in records:
        masks[rec_i] = np.array([
            (rec_i, i) in selected for i in range(len(rec["y"]))
        ], dtype=bool)
    return masks


def normalize(train_x, val_x, test_x):
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std, mean, std


def make_event_predictions(model, records, held_out_i, modality, mean, std):
    rec = records[held_out_i]
    x = rec["x_emg"]
    if modality == "FUSION":
        frame = pd.read_csv(rec["csv"], usecols=["accel_x", "accel_y", "accel_z"])
        acc = frame.to_numpy(dtype=np.float32)
        acc_windows = []
        for event in rec["events"]:
            a, b = event["sample_start"], event["sample_stop"]
            for start in range(a, b - WINDOW + 1, HOP):
                acc_windows.append(acc[start:start + WINDOW])
        x = np.concatenate([x, np.stack(acc_windows)], axis=2)
    p = model.predict((x - mean) / std, batch_size=32, verbose=0)
    pred_idx = np.argmax(p, axis=1)
    idx_to_class = np.asarray(CLASSES)
    rows = []
    for event in rec["events"]:
        a, b = event["window_start"], event["window_stop"]
        pred = max(CLASSES, key=lambda c: int(np.sum(idx_to_class[pred_idx[a:b]] == c)))
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
        "events": len(rows),
        "event_accuracy": float(accuracy_score(truth, pred)),
        "event_macro_f1": float(f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0)),
        "gesture_events": len(gestures),
        "gesture_accuracy": float(accuracy_score([r["true_label"] for r in gestures], [r["predicted_label"] for r in gestures])),
        "tilt_controls": len(controls),
        "tilt_false_commands": sum(r["predicted_label"] != "rest" for r in controls),
        "tilt_false_command_rate": sum(r["predicted_label"] != "rest" for r in controls) / max(1, len(controls)),
        "per_class": {
            c: {"correct": int(matrix[i, i]), "total": int(matrix[i].sum()),
                "recall": float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else 0.0}
            for i, c in enumerate(CLASSES)
        },
        "confusion_labels": CLASSES,
        "confusion_true_rows_pred_columns": matrix.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", action="append", nargs=3, metavar=("NAME", "CSV", "CUE"), required=True)
    ap.add_argument("--out-dir", type=Path, default=HERE / "cnn_lstm_sessions")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if sorted(s[0] for s in args.session) != SESSIONS:
        raise SystemExit(f"Supply exactly these four session names: {', '.join(SESSIONS)}")

    ev = load_evaluator()
    records = []
    for name, csv_arg, cue_arg in sorted(args.session, key=lambda x: SESSIONS.index(x[0])):
        rec = load_session(ev, name, Path(csv_arg).resolve(), Path(cue_arg).resolve())
        # Add exact sample bounds for aligned six-channel input windows.
        cue = json.loads(Path(cue_arg).read_text())
        intervals = ev.get_action_intervals(cue, len(pd.read_csv(csv_arg, usecols=["ch1"])))
        for event, interval in zip(rec["events"], intervals):
            event["sample_start"], event["sample_stop"] = interval["start"], interval["stop"]
        records.append(rec)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}
    pooled_rows = {"EMG": [], "FUSION": []}
    for modality in ("EMG", "FUSION"):
        fold_metrics = {}
        for held_out_i, held_out in enumerate(SESSIONS):
            train_ids = {i for i in range(len(records)) if i != held_out_i}
            # Select complete cues for internal validation. These are drawn
            # only from the three training sessions; the held-out wearing is
            # untouched until event-level scoring.
            val_mask = make_validation_split([(i, records[i]) for i in train_ids], SEED + held_out_i)
            train_x_parts, train_y_parts, val_x_parts, val_y_parts = [], [], [], []
            for i in train_ids:
                rec = records[i]
                x = rec["x_emg"]
                if modality == "FUSION":
                    frame = pd.read_csv(rec["csv"], usecols=["accel_x", "accel_y", "accel_z"])
                    acc = frame.to_numpy(dtype=np.float32)
                    aw = []
                    for e in rec["events"]:
                        for start in range(e["sample_start"], e["sample_stop"] - WINDOW + 1, HOP):
                            aw.append(acc[start:start + WINDOW])
                    x = np.concatenate([x, np.stack(aw)], axis=2)
                vmask = val_mask[i]
                if vmask.any():
                    train_x_parts.append(x[~vmask]); train_y_parts.append(rec["y"][~vmask])
                    val_x_parts.append(x[vmask]); val_y_parts.append(rec["y"][vmask])
                else:
                    train_x_parts.append(x); train_y_parts.append(rec["y"])
            x_train = np.concatenate(train_x_parts); y_train = np.concatenate(train_y_parts)
            x_val = np.concatenate(val_x_parts); y_val = np.concatenate(val_y_parts)
            x_test = records[held_out_i]["x_emg"]
            if modality == "FUSION":
                rec = records[held_out_i]
                frame = pd.read_csv(rec["csv"], usecols=["accel_x", "accel_y", "accel_z"])
                acc = frame.to_numpy(dtype=np.float32)
                aw = []
                for e in rec["events"]:
                    for start in range(e["sample_start"], e["sample_stop"] - WINDOW + 1, HOP):
                        aw.append(acc[start:start + WINDOW])
                x_test = np.concatenate([x_test, np.stack(aw)], axis=2)
            x_train, x_val, x_test, mean, std = normalize(x_train, x_val, x_test)
            keras.utils.set_random_seed(SEED + held_out_i)
            from model_architectures import build_model
            model = build_model("cnn_lstm", WINDOW, x_train.shape[2], len(CLASSES))
            y_train_oh = keras.utils.to_categorical([CLASSES.index(y) for y in y_train], len(CLASSES))
            y_val_oh = keras.utils.to_categorical([CLASSES.index(y) for y in y_val], len(CLASSES))
            callbacks = [
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
            ]
            history = model.fit(x_train, y_train_oh, validation_data=(x_val, y_val_oh),
                                epochs=args.epochs, batch_size=args.batch_size,
                                callbacks=callbacks, verbose=0, shuffle=True)
            rows = make_event_predictions(model, records, held_out_i, modality, mean, std)
            pooled_rows[modality].extend(rows)
            fold_metrics[held_out] = {
                "train_sessions": [SESSIONS[i] for i in sorted(train_ids)],
                "train_windows": int(len(y_train)), "validation_windows": int(len(y_val)),
                "test_windows": int(len(records[held_out_i]["y"])),
                "epochs_run": len(history.history["loss"]),
                "best_validation_loss": float(min(history.history["val_loss"])),
                "metrics": summarize(rows), "predictions": rows,
            }
            print(f"{modality} held out {held_out}: {fold_metrics[held_out]['metrics']['gesture_accuracy']:.3f} gesture accuracy; "
                  f"{fold_metrics[held_out]['metrics']['tilt_false_commands']}/40 tilt false commands", flush=True)
        all_results[modality] = {"folds": fold_metrics, "pooled": summarize(pooled_rows[modality])}

    # Once the fold screen is complete, refit the better modality on all four
    # sessions using 20% complete-cue validation for early stopping. This is
    # a personalized development model; it has no independent accuracy score.
    selected = max(("EMG", "FUSION"), key=lambda m: (
        all_results[m]["pooled"]["gesture_accuracy"],
        -all_results[m]["pooled"]["tilt_false_command_rate"],
    ))
    train_ids = set(range(len(records)))
    val_mask = make_validation_split([(i, records[i]) for i in train_ids], SEED + 100)
    train_x_parts, train_y_parts, val_x_parts, val_y_parts = [], [], [], []
    for i, rec in enumerate(records):
        x = rec["x_emg"]
        if selected == "FUSION":
            frame = pd.read_csv(rec["csv"], usecols=["accel_x", "accel_y", "accel_z"])
            acc = frame.to_numpy(dtype=np.float32)
            aw = []
            for e in rec["events"]:
                for start in range(e["sample_start"], e["sample_stop"] - WINDOW + 1, HOP):
                    aw.append(acc[start:start + WINDOW])
            x = np.concatenate([x, np.stack(aw)], axis=2)
        vm = val_mask[i]
        train_x_parts.append(x[~vm]); train_y_parts.append(rec["y"][~vm])
        val_x_parts.append(x[vm]); val_y_parts.append(rec["y"][vm])
    x_train = np.concatenate(train_x_parts); y_train = np.concatenate(train_y_parts)
    x_val = np.concatenate(val_x_parts); y_val = np.concatenate(val_y_parts)
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True); std[std == 0] = 1.0
    keras.utils.set_random_seed(SEED + 100)
    from model_architectures import build_model
    final_model = build_model("cnn_lstm", WINDOW, x_train.shape[2], len(CLASSES))
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)]
    y_train_oh = keras.utils.to_categorical([CLASSES.index(y) for y in y_train], len(CLASSES))
    y_val_oh = keras.utils.to_categorical([CLASSES.index(y) for y in y_val], len(CLASSES))
    hist = final_model.fit((x_train - mean) / std, y_train_oh,
                           validation_data=((x_val - mean) / std, y_val_oh),
                           epochs=args.epochs, batch_size=args.batch_size,
                           callbacks=callbacks, verbose=0, shuffle=True)
    model_path = out_dir / "cnn_lstm_all_sessions.keras"
    final_model.save(model_path)
    metadata = {
        "status": "personalized development candidate; no independent test score",
        "architecture": "supplier CNN-LSTM",
        "training_sessions": SESSIONS,
        "selected_modality": selected,
        "modality_selection_rule": "highest pooled held-session gesture accuracy; false tilt-control rate is reported separately and remains a gate for use",
        "classes": CLASSES,
        "input_channels": ["ch1", "ch2", "ch3"] + (["accel_x", "accel_y", "accel_z"] if selected == "FUSION" else []),
        "window_samples": WINDOW, "window_ms_nominal": WINDOW,
        "hop_samples": HOP, "hop_ms_nominal": HOP,
        "training_cue_validation": "20% of complete prompted intervals held out for early stopping; scaler statistics fit on remaining intervals only",
        "final_training_windows": int(len(y_train)),
        "internal_validation_windows": int(len(y_val)),
        "normalization_mean": mean.reshape(-1).tolist(),
        "normalization_std": std.reshape(-1).tolist(),
        "cross_session_screen": all_results,
        "final_fit_epochs": len(hist.history["loss"]),
        "model_path": str(model_path),
        "session_hashes": {r["name"]: {"csv": r["csv_sha256"], "cue": r["cue_sha256"]} for r in records},
        "limits": [
            "All four sessions have now been used for development; none is a future independent test.",
            "The four-fold candidate screen selected between EMG-only and fused inputs, so pooled CV is a development estimate and somewhat optimistic.",
            "The cue-vote metrics do not measure movement-onset latency or ordinary-activity false-command rate.",
        ],
    }
    report_path = out_dir / "cnn_lstm_session_loso.json"
    report_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"selected_modality": selected, "pooled": {m: all_results[m]["pooled"] for m in all_results},
                      "model": str(model_path), "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
