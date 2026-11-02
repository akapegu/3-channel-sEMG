"""Create a provisional, human-reviewable activity annotation sheet.

This script never edits source recordings and never assigns gesture labels.
The thresholded intervals are only candidates for visual/user review.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FILES = ["swipe_down.csv", "swipe_left.csv", "swipe_right.csv", "swipe_top.csv"]
CHANNELS = ["Channel1", "Channel2", "Channel3"]
FS_HZ_ASSUMED = 1000
ENV_WINDOW = 200
ENV_HOP = 50
SMOOTH_BINS = 3
MAX_GAP_BINS = 2
MIN_ACTIVE_BINS = 2


def load(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    x = np.asarray([[float(r[c]) for c in CHANNELS] for r in rows], dtype=float)
    counters = np.asarray([int(r["Counter"]) for r in rows], dtype=np.int64)
    return x, counters


def runs(mask: np.ndarray):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)  # exclusive
    return list(zip(starts.tolist(), ends.tolist()))


def main():
    env_fig, axes = plt.subplots(4, 1, figsize=(15, 12), constrained_layout=True)
    colors = ["#2463a6", "#dc7b23", "#29945b"]
    all_rows = []
    summaries = {}

    for ax, filename in zip(axes, FILES):
        label = filename.removeprefix("swipe_").removesuffix(".csv")
        path = ROOT / "recordings" / filename
        x, counter = load(path)
        env = []
        times = []
        for end in range(ENV_WINDOW, len(x) + 1, ENV_HOP):
            z = x[end - ENV_WINDOW:end]
            z = z - np.mean(z, axis=0, keepdims=True)
            env.append(np.sqrt(np.mean(z * z, axis=0)))
            times.append((end - ENV_WINDOW / 2) / FS_HZ_ASSUMED)
        env = np.asarray(env)
        times = np.asarray(times)

        # Normalize each envelope by its robust spread so a single high-gain
        # channel does not dominate. Threshold is heuristic, not calibrated.
        med = np.median(env, axis=0)
        mad = 1.4826 * np.median(np.abs(env - med), axis=0)
        scale = np.maximum(mad, np.maximum(med * 0.05, 1e-9))
        score = np.max((env - med) / scale, axis=1)
        kernel = np.ones(SMOOTH_BINS) / SMOOTH_BINS
        smooth_score = np.convolve(score, kernel, mode="same")
        threshold = 2.5
        active = smooth_score >= threshold

        # Merge brief below-threshold gaps to avoid splitting one burst.
        spans = runs(active)
        for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
            if s2 - e1 <= MAX_GAP_BINS:
                active[e1:s2] = True
        candidates = [(s, e) for s, e in runs(active) if e - s >= MIN_ACTIVE_BINS]

        for ch in range(3):
            ax.plot(times, env[:, ch], lw=1, color=colors[ch], label=f"CH{ch+1}")
        for i, (s, e) in enumerate(candidates, 1):
            start_s = max(0.0, times[s] - ENV_HOP / FS_HZ_ASSUMED / 2)
            end_s = min((len(x) - 1) / FS_HZ_ASSUMED,
                        times[min(e - 1, len(times) - 1)] + ENV_HOP / FS_HZ_ASSUMED / 2)
            ax.axvspan(start_s, end_s, color="#f0c419", alpha=.22)
            cid = f"{label.upper()}-{i:02d}"
            all_rows.append({
                "candidate_id": cid,
                "source_file": filename,
                "candidate_start_s_nominal": f"{start_s:.3f}",
                "candidate_end_s_nominal": f"{end_s:.3f}",
                "candidate_start_sample_assumed_1000sps": round(start_s * FS_HZ_ASSUMED),
                "candidate_end_sample_assumed_1000sps_exclusive": round(end_s * FS_HZ_ASSUMED),
                "suggested_class": "",
                "human_confirmed_active_swipe": "",
                "human_corrected_start_s": "",
                "human_corrected_end_s": "",
                "confidence_high_medium_low": "",
                "notes": "",
            })
        ax.set_title(f"{label}: 200 ms RMS, yellow = threshold candidate only ({len(candidates)})")
        ax.set_ylabel("Processed value RMS")
        ax.grid(alpha=.2)
        if label == "down":
            ax.legend(ncol=3, fontsize=8)
        ax.set_xlabel("Time (s; nominal 1000 SPS, no timestamps in source)")
        transitions = np.diff(counter) % 256
        summaries[label] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": int(len(x)),
            "nominal_duration_s_assuming_1000_sps": float((len(x)-1)/FS_HZ_ASSUMED),
            "nonunit_counter_transitions": int(np.sum(transitions != 1)),
            "candidate_count": len(candidates),
        }

    env_fig.savefig(OUT / "activity_candidate_review.png", dpi=160)
    plt.close(env_fig)
    csv_path = OUT / "activity_annotation_review.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]) if all_rows else [])
        writer.writeheader()
        writer.writerows(all_rows)
    manifest = {
        "purpose": "Human review aid only; no gesture class labels are assigned.",
        "threshold_method": {
            "envelope": f"mean-centered RMS over {ENV_WINDOW} samples, updated every {ENV_HOP} samples",
            "nominal_sampling_rate_hz_assumed": FS_HZ_ASSUMED,
            "score": "maximum across channels of (channel RMS - channel median) / robust MAD scale; scale= max(1.4826*MAD, 0.05*median, 1e-9)",
            "threshold": 2.5,
            "postprocessing": f"{SMOOTH_BINS}-bin moving average; merge gaps <= {MAX_GAP_BINS} bins; keep intervals >= {MIN_ACTIVE_BINS} bins",
        },
        "limitations": [
            "The exports contain no cue/event timestamps; candidate activity does not prove intentional swipe movement.",
            "The 1000 SPS rate is nominal because the exports have no sample timestamps.",
            "Thresholds may miss weak motion or include artifacts; review all four full traces and correct boundaries manually.",
            "Candidate class is intentionally blank; filename class applies only after user confirms the movement.",
        ],
        "recordings": summaries,
        "artifacts": [csv_path.name, "activity_candidate_review.png"],
    }
    (OUT / "activity_annotation_review.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
