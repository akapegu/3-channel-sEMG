"""Profile repeated fist-clench browser export and produce a replay trigger trace."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "recordings/NPG-Lite-Band-3CH-27-BC-D1-ChordsWeb-20261101-044735.csv"
OUT = Path(__file__).resolve().parent
FS = 1000.0  # nominal sample rate; this CSV has no timestamps
WINDOW = 100
HOP = 50
BASELINE_SECONDS = 2.0
MIN_HIGH = 0.05
HIGH_MAD_MULTIPLIER = 8.0
LOW_RATIO = 0.55
MIN_HIGH_WINDOWS = 2
MIN_LOW_WINDOWS = 3
REFRACTORY_SECONDS = 0.40


def detect_events(times, envelope, high, low):
    events = []
    above = below = 0
    armed = True
    last_event = -1e9
    for i, value in enumerate(envelope):
        if value >= high:
            above += 1
            below = 0
            if (armed and above >= MIN_HIGH_WINDOWS
                    and times[i] - last_event >= REFRACTORY_SECONDS):
                events.append(float(times[i]))
                last_event = times[i]
                armed = False
        else:
            above = 0
            if value < low:
                below += 1
            else:
                below = 0
            if below >= MIN_LOW_WINDOWS:
                armed = True
    return events


def main():
    with SOURCE.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    x = np.array([[float(r[f"Channel{i}"]) for i in (1, 2, 3)] for r in rows], dtype=float)
    counters = np.array([int(r["Counter"]) for r in rows], dtype=np.int64)
    if len(x) < WINDOW or x.shape[1] != 3:
        raise ValueError("Expected a non-empty three-channel browser export")

    transitions = np.diff(counters) % 256
    env_values, env_times = [], []
    for start in range(0, len(x) - WINDOW + 1, HOP):
        block = x[start:start + WINDOW, 0]
        centered = block - block.mean()
        env_values.append(float(np.sqrt(np.mean(centered ** 2))))
        env_times.append((start + (WINDOW - 1) / 2) / FS)
    env = np.asarray(env_values)
    times = np.asarray(env_times)

    baseline = env[times < BASELINE_SECONDS]
    if len(baseline) < 10:
        raise ValueError("Need at least 0.5 seconds of initial relaxed baseline")
    median = float(np.median(baseline))
    scaled_mad = float(1.4826 * np.median(np.abs(baseline - median)))
    high = max(MIN_HIGH, median + HIGH_MAD_MULTIPLIER * scaled_mad)
    low = LOW_RATIO * high
    events = detect_events(times, env, high, low)

    with (OUT / "channel1_envelope.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["approx_time_s", "channel1_100ms_rms", "high_threshold", "low_threshold", "event"])
        event_bins = {int(np.argmin(np.abs(times - e))) for e in events}
        w.writerows((float(t), float(v), high, low, int(i in event_bins))
                    for i, (t, v) in enumerate(zip(times, env)))

    raw_t = np.arange(len(x)) / FS
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True, layout="constrained")
    colors = ("#2563a6", "#cf7623", "#38865b")
    for ch in range(3):
        axes[ch].plot(raw_t, x[:, ch], color=colors[ch], lw=0.45)
        axes[ch].set_ylabel(f"Channel {ch + 1}\nprocessed")
        axes[ch].grid(alpha=0.22)
    axes[3].plot(times, env, color=colors[0], lw=1.0, label="100 ms mean-centered RMS")
    axes[3].axhline(high, color="#a33", lw=1, label=f"trigger threshold {high:.3f}")
    axes[3].axhline(low, color="#777", lw=0.8, ls="--", label=f"release threshold {low:.3f}")
    for event in events:
        axes[3].axvline(event, color="#a33", alpha=0.22, lw=0.8)
    axes[3].set_ylabel("Channel 1\nRMS")
    axes[3].set_xlabel("Approximate time from first sample (s; nominal 1000 samples/s)")
    axes[3].grid(alpha=0.22)
    axes[3].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0].set_title("Repeated fist signal: one-channel event detector (candidate triggers marked below)")
    fig.savefig(OUT / "repeated_fist_detector.png", dpi=150)
    plt.close(fig)

    report = {
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "columns": columns,
        "sample_count": int(len(rows)),
        "nominal_sample_rate_hz": FS,
        "approx_duration_s": len(rows) / FS,
        "counter_first_last": [int(counters[0]), int(counters[-1])],
        "counter_nonunit_transitions": int(np.sum(transitions != 1)),
        "counter_wraps": int(np.sum(np.diff(counters) < 0)),
        "all_values_finite": bool(np.isfinite(x).all()),
        "detector": {
            "input": "Channel1 processed browser values",
            "envelope": {"window_ms": 100, "hop_ms": 50, "mean_centered": True},
            "initial_relaxed_baseline_assumption_s": BASELINE_SECONDS,
            "baseline_median": median,
            "baseline_scaled_mad": scaled_mad,
            "high_threshold": high,
            "low_threshold": low,
            "high_consecutive_windows": MIN_HIGH_WINDOWS,
            "low_consecutive_windows_to_rearm": MIN_LOW_WINDOWS,
            "refractory_s": REFRACTORY_SECONDS,
            "candidate_event_count": len(events),
            "candidate_event_times_s": [round(t, 3) for t in events],
        },
        "limitations": [
            "Seconds are approximate because the CSV has no timestamps; nominal 1000 samples/s is assumed.",
            "Thresholds are an offline prototype for this processed export, not validated for live use.",
            "Candidate detector events are not confirmed intended clench counts or false-positive measurements.",
            "Processed values are not raw ADC counts or calibrated physical units."
        ]
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
