"""Descriptively audit held-out cue-level predictions across Sessions 1–4.

This script reads the already-scored locked-test report and preserved processed
CSV/cue timeline. It does not fit or change a model, thresholds, labels, or data.
Signal summaries are in the processed recording's arbitrary scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CSV = ROOT / "supplier_toolkit/adapted/training_data/subject_01/session4/continuous/swipe4_control_trial1_20261104-003622_merged_ECF6.csv"
CUE = ROOT / "raw_packets/1793732227632908000_dev1.session_cues.json"
SCORE = HERE / "session4_locked_test.json"
LOSO = HERE / "session1_3_leave_one_session_out.json"
OUT = HERE / "cross_session_error_audit"
EMG = ["ch1", "ch2", "ch3"]
ACC = ["accel_x", "accel_y", "accel_z"]
SESSION_FILES = {
    "session1": (ROOT / "supplier_toolkit/adapted/training_data/abinash_swipe4_robust/session1/continuous/swipe4_control_trial1_20261103-223103_merged_ECF6.csv",
                 ROOT / "supplier_toolkit/adapted/raw_packets/1793724704866028000_dev1.session_cues.json"),
    "session2": (ROOT / "supplier_toolkit/adapted/training_data/subject_01/session2/continuous/swipe4_control_trial1_20261103-231031_merged_ECF6.csv",
                 ROOT / "raw_packets/1793727101604536000_dev1.session_cues.json"),
    "session3": (ROOT / "supplier_toolkit/adapted/training_data/subject_01/session3/continuous/swipe4_control_trial1_20261103-235957_merged_ECF6.csv",
                 ROOT / "raw_packets/1793730071821897000_dev1.session_cues.json"),
    "session4": (CSV, CUE),
}


def action_intervals(cue: dict, n: int) -> list[dict]:
    starts: dict[int, dict] = {}
    rows = []
    for event in cue["events"]:
        if event.get("phase") != "ACTION" or event.get("event") not in {"label_start", "label_stop"}:
            continue
        idx = int(event["phase_index"])
        if event["event"] == "label_start":
            starts[idx] = event
            continue
        start = starts.pop(idx)
        a, b = int(start["sample_index"]), int(event["sample_index"])
        if not (0 <= a < b <= n):
            raise ValueError(f"Invalid action interval [{a},{b})")
        rows.append({"start": a, "stop": b, "label": start["label"],
                     "trial_type": start.get("trial_type"),
                     "direction": start.get("direction"),
                     "trial_index": int(start["trial_index"])})
    if starts:
        raise ValueError("Unclosed action cue(s)")
    return sorted(rows, key=lambda row: row["start"])


def centered_rms(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.mean(x, axis=0, keepdims=True)
    return np.sqrt(np.mean(x * x, axis=0))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    score4 = json.loads(SCORE.read_text())
    loso = json.loads(LOSO.read_text())
    if score4.get("no_fitting_or_tuning_on_session4") is not True:
        raise ValueError("Expected the preserved locked-test scoring report")
    rows = []
    prediction_by_session = {s: loso["models"]["EXTRA_TREES_FUSION_400ms"]["folds"][s]["metrics"]["event_predictions"] for s in ("session1", "session2", "session3")}
    prediction_by_session["session4"] = score4["metrics"]["event_predictions"]
    for session, (csv_path, cue_path) in SESSION_FILES.items():
        df = pd.read_csv(csv_path)
        cue = json.loads(cue_path.read_text())
        if cue.get("status") != "complete" or cue.get("session") != session:
            raise ValueError(f"Expected complete {session} cue timeline")
        intervals = action_intervals(cue, len(df))
        predictions = prediction_by_session[session]
        if len(intervals) != len(predictions) or len(intervals) != 120:
            raise ValueError(f"{session} cues do not match saved held-out predictions")
        for iv, pred in zip(intervals, predictions):
            if iv["trial_index"] != pred["trial_index"]:
                raise ValueError(f"{session} prediction and cue order/index mismatch")
            seg = df.iloc[iv["start"]:iv["stop"]]
            emg = seg[EMG].to_numpy(dtype=np.float64)
            acc = seg[ACC].to_numpy(dtype=np.float64)
            row = {"session": session, **iv, "true_label": pred["true_label"],
                   "predicted_label": pred["predicted_label"],
                   "correct": bool(pred["correct"]),
                   "duration_s": (iv["stop"] - iv["start"]) / 1000.0}
            rms = centered_rms(emg)
            for j, name in enumerate(EMG):
                row[f"{name}_rms"] = float(rms[j])
                row[f"{name}_mav"] = float(np.mean(np.abs(emg[:, j] - np.mean(emg[:, j]))))
                # Three consecutive 500 ms bins show signal localization within each cue.
                for bin_no in range(3):
                    lo, hi = round(bin_no * len(emg) / 3), round((bin_no + 1) * len(emg) / 3)
                    row[f"{name}_rms_bin{bin_no+1}"] = float(centered_rms(emg[lo:hi])[j])
            acc_mag = np.linalg.norm(acc, axis=1)
            row["accel_magnitude_std"] = float(np.std(acc_mag))
            row["accel_magnitude_range"] = float(np.max(acc_mag) - np.min(acc_mag))
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT / "all_sessions_cue_signal_summary.csv", index=False)
    summary = {
        "analysis": "descriptive cue-level audit; no model fitting or tuning",
        "sources": {s: {"csv": str(pair[0].relative_to(ROOT)), "cue": str(pair[1].relative_to(ROOT))}
                    for s, pair in SESSION_FILES.items()},
        "prediction_provenance": {
            "session1_to_session3": "saved leave-one-session-out ExtraTrees fusion predictions; each evaluated session was excluded from that fold's fit",
            "session4": "saved locked test predictions; model fit on Sessions 1–3; scored once",
        },
        "rows": len(out_df),
        "signal_scale": "processed EMG and accelerometer values; arbitrary scale, not calibrated physical units",
        "measurements": "per-cue centered EMG RMS/MAV, 500 ms bin RMS, and accelerometer magnitude variability",
        "groups": {},
        "limits": [
            "Sessions 1–3 predictions are leave-one-session-out estimates; Session 4 was already consumed as the final test. This cross-session descriptive audit cannot support an unbiased new score or model selection.",
            "The 1.5 s visual cue is not an independently measured muscle-action onset; its three bins can mix preparation, action, and release.",
            "One participant and one re-wearing; cue-level comparisons are small descriptive groups.",
            "Accelerometer magnitude is only a movement summary and does not establish that tilt caused a prediction.",
        ],
    }
    measure_cols = [f"{c}_rms" for c in EMG] + ["accel_magnitude_std", "accel_magnitude_range"]
    def group_summary(data: pd.DataFrame, by: list[str]) -> dict:
        result = {}
        for key, group in data.groupby(by, dropna=False, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            name = " / ".join(f"{col}={value}" for col, value in zip(by, key))
            result[name] = {"n": int(len(group)), "median": {
                col: round(float(group[col].median()), 5) for col in measure_cols}}
        return result

    summary["groups"]["gestures_by_session_true_and_predicted_label"] = group_summary(
        out_df[out_df.trial_type == "gesture"], ["session", "true_label", "predicted_label"])
    summary["groups"]["tilt_controls_by_session_direction_and_predicted_label"] = group_summary(
        out_df[out_df.trial_type == "tilt_only_control"], ["session", "direction", "predicted_label"])
    summary["prediction_counts"] = {
        "per_session": {},
        "confusion_labels": ["rest", "down", "left", "right", "top"],
    }
    for session, group in out_df.groupby("session", sort=True):
        matrix = []
        labels = summary["prediction_counts"]["confusion_labels"]
        for truth in labels:
            matrix.append([int(((group.true_label == truth) & (group.predicted_label == guess)).sum()) for guess in labels])
        gestures = group[group.trial_type == "gesture"]
        tilt = group[group.trial_type == "tilt_only_control"]
        summary["prediction_counts"]["per_session"][session] = {
            "event_accuracy": float(group.correct.mean()),
            "gesture_correct": int(gestures.correct.sum()), "gesture_count": int(len(gestures)),
            "gesture_accuracy": float(gestures.correct.mean()),
            "tilt_false_commands": int((tilt.predicted_label != "rest").sum()),
            "tilt_control_count": int(len(tilt)),
            "confusion_true_rows_pred_columns": matrix,
        }
    fitted = joblib.load(HERE / "model_session123" / "extra_trees_fusion_400ms.joblib")
    feature_names = ([f"log1p_{stat}_{ch}" for stat in ("rms", "mav", "wl") for ch in EMG]
                     + [f"accel_{stat}_{axis}" for stat in ("mean", "std") for axis in ("x", "y", "z")])
    importances = dict(zip(feature_names, map(float, fitted.feature_importances_)))
    summary["frozen_model_feature_importance"] = {
        "values": {name: round(value, 6) for name, value in sorted(importances.items(), key=lambda item: -item[1])},
        "emg_feature_importance_sum": round(sum(value for name, value in importances.items() if name.startswith("log1p_")), 6),
        "accelerometer_feature_importance_sum": round(sum(value for name, value in importances.items() if name.startswith("accel_")), 6),
        "warning": "ExtraTrees impurity importance is a model-internal split-use summary, not a causal measure or proof of modality dependence.",
    }
    (OUT / "cross_session_error_audit.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Write a standalone SVG so the audit does not require optional plotting packages.
    gesture = out_df[(out_df.session == "session4") & (out_df.trial_type == "gesture")]
    class_order = ["down", "left", "right", "top"]
    width, height = 1500, 720
    panel_w, top, bottom = 475, 130, 610
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="750" y="34" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="bold">Session 4 held-out gesture cues: EMG RMS by frozen-model outcome</text>',
             '<text x="750" y="62" text-anchor="middle" font-family="sans-serif" font-size="14">Blue = correctly classified; orange = missed. Each dot is one 1.5 s visual cue; horizontal bars show medians.</text>']
    for panel, ch in enumerate(EMG):
        x0 = 65 + panel * 490
        all_values = gesture[f"{ch}_rms"].to_numpy()
        ymax = max(float(np.max(all_values)) * 1.08, 1e-6)
        parts.append(f'<text x="{x0+220}" y="95" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold">{ch.upper()} cue RMS</text>')
        # Y axis and horizontal guides.
        for tick in range(6):
            val = ymax * tick / 5
            y = bottom - (bottom-top) * tick / 5
            parts.append(f'<line x1="{x0+48}" y1="{y:.1f}" x2="{x0+455}" y2="{y:.1f}" stroke="#dddddd"/>')
            parts.append(f'<text x="{x0+40}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{val:.1f}</text>')
        parts.append(f'<line x1="{x0+48}" y1="{top}" x2="{x0+48}" y2="{bottom}" stroke="#333"/>')
        for cls_i, cls in enumerate(class_order):
            for result_i, correct in enumerate((True, False)):
                cat_i = cls_i * 2 + result_i
                xcenter = x0 + 78 + cat_i * 48
                col = "#277da1" if correct else "#f3722c"
                data = gesture[(gesture.true_label == cls) & (gesture.correct == correct)]
                vals = data[f"{ch}_rms"].to_numpy()
                for event_i, val in enumerate(vals):
                    # Deterministic small horizontal jitter, no randomness in the audit.
                    jitter = ((event_i * 17 + cat_i * 7) % 19 - 9) * 1.3
                    y = bottom - (bottom-top) * float(val) / ymax
                    parts.append(f'<circle cx="{xcenter+jitter:.1f}" cy="{y:.1f}" r="4" fill="{col}" fill-opacity="0.78"/>')
                if len(vals):
                    med_y = bottom - (bottom-top) * float(np.median(vals)) / ymax
                    parts.append(f'<line x1="{xcenter-16}" y1="{med_y:.1f}" x2="{xcenter+16}" y2="{med_y:.1f}" stroke="{col}" stroke-width="3"/>')
                tag = "OK" if correct else "MISS"
                parts.append(f'<text x="{xcenter}" y="{bottom+18}" text-anchor="middle" font-family="sans-serif" font-size="9">{cls.upper()}</text>')
                parts.append(f'<text x="{xcenter}" y="{bottom+31}" text-anchor="middle" font-family="sans-serif" font-size="8">{tag} n={len(vals)}</text>')
        parts.append(f'<text x="{x0+8}" y="{(top+bottom)/2}" text-anchor="middle" transform="rotate(-90 {x0+8} {(top+bottom)/2})" font-family="sans-serif" font-size="12">Processed signal scale</text>')
    parts.append('</svg>')
    plot_path = OUT / "session4_cue_rms_by_outcome.svg"
    plot_path.write_text("\n".join(parts) + "\n")
    print(json.dumps({"rows": len(out_df), "groups": len(summary["groups"]),
                      "per_session": summary["prediction_counts"]["per_session"],
                      "output_dir": str(OUT), "plot": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
