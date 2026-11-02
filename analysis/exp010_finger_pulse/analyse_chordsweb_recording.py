"""Read-only profile and plot of the 2026-11-01 ChordsWeb export.

This export has no timestamps or event labels. The x-axis uses the nominal
1000 samples/s rate only; no onset detection or gesture classification is
performed here.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "recordings/NPG-Lite-Band-3CH-27-BC-D1-ChordsWeb-20261101-043641.csv"
OUT = Path(__file__).resolve().parent
FS_NOMINAL = 1000.0
CHANNELS = ("Channel1", "Channel2", "Channel3")


def main():
    with SOURCE.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    counter = np.array([int(row["Counter"]) for row in rows], dtype=np.int64)
    x = np.array([[float(row[name]) for name in CHANNELS] for row in rows], dtype=float)
    if len(x) == 0 or x.shape[1] != 3:
        raise ValueError("Expected a non-empty three-channel CSV")

    transitions = (np.diff(counter) % 256).astype(int)
    stats = {}
    for i, name in enumerate(CHANNELS):
        q = np.quantile(x[:, i], [0.01, 0.5, 0.99])
        stats[name] = {
            "min": float(np.min(x[:, i])),
            "p01": float(q[0]),
            "median": float(q[1]),
            "p99": float(q[2]),
            "max": float(np.max(x[:, i])),
            "mean": float(np.mean(x[:, i])),
            "std": float(np.std(x[:, i])),
        }

    # A centered 100 ms RMS window, advanced every 50 ms. This makes short
    # activations easier to see without assigning gesture labels.
    win = 100
    hop = 50
    envelope_t = []
    envelope = []
    for start in range(0, len(x) - win + 1, hop):
        block = x[start:start + win]
        centered = block - block.mean(axis=0, keepdims=True)
        envelope.append(np.sqrt(np.mean(centered ** 2, axis=0)))
        envelope_t.append((start + (win - 1) / 2) / FS_NOMINAL)
    envelope = np.asarray(envelope)
    envelope_t = np.asarray(envelope_t)

    # Conservative CH1 peaks retained as a reference to the first plot. These
    # are exploratory event markers, not labels.
    peaks, peak_props = find_peaks(envelope[:, 0], height=0.04, distance=6,
                                   prominence=0.018)
    with (OUT / "candidate_pulse_peaks.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["approx_time_s", "channel1_100ms_rms", "prominence", "label"])
        writer.writerows((float(envelope_t[i]), float(envelope[i, 0]),
                           float(peak_props["prominences"][j]), "unlabeled_candidate")
                          for j, i in enumerate(peaks))

    # Lower per-channel thresholds expose weaker peaks omitted by the CH1-only
    # pass. Thresholds reflect channel scale, so these must not be interpreted
    # as calibrated or validated contraction-onset criteria.
    peak_thresholds = (0.015, 0.008, 0.012)
    per_channel_peaks = []
    with (OUT / "candidate_pulse_peaks_multichannel.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["channel", "approx_time_s", "100ms_rms", "threshold", "prominence", "label"])
        for ch, threshold in enumerate(peak_thresholds):
            idx, props = find_peaks(envelope[:, ch], height=threshold, distance=6,
                                    prominence=threshold * 0.45)
            per_channel_peaks.append((idx, props))
            writer.writerows((ch + 1, float(envelope_t[i]), float(envelope[i, ch]),
                              threshold, float(props["prominences"][j]), "unlabeled_candidate")
                             for j, i in enumerate(idx))

    with (OUT / "rms_envelope_100ms.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["approx_time_s", "rms_channel1", "rms_channel2", "rms_channel3"])
        writer.writerows((t, *v) for t, v in zip(envelope_t, envelope))

    fig, axes = plt.subplots(6, 1, figsize=(14, 13), sharex=True, layout="constrained")
    colors = ("#1f77b4", "#d17a22", "#38865b")
    t = np.arange(len(x)) / FS_NOMINAL
    for ch, name in enumerate(CHANNELS):
        axes[ch].plot(t, x[:, ch], color=colors[ch], lw=0.45)
        axes[ch].set_ylabel(f"{name}\nprocessed")
        axes[ch].grid(alpha=0.22)
        axes[ch + 3].plot(envelope_t, envelope[:, ch], color=colors[ch], lw=0.9)
        axes[ch + 3].set_ylabel(f"{name}\n100 ms RMS")
        axes[ch + 3].grid(alpha=0.22)
    for ch, (idx, _) in enumerate(per_channel_peaks):
        axes[ch + 3].scatter(envelope_t[idx], envelope[idx, ch], marker="x", s=20,
                             color=colors[ch], label=f"candidate peaks CH{ch+1}; unlabeled")
        axes[ch + 3].legend(loc="upper right", fontsize=8)
    axes[0].set_title("ChordsWeb finger-pulse recording — no event markers supplied")
    axes[2].set_ylabel("Channel3\nprocessed")
    axes[-1].set_xlabel("Approximate time from first sample (s; nominal 1000 samples/s)")
    fig.savefig(OUT / "finger_pulse_overview.png", dpi=150)
    plt.close(fig)

    report = {
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "columns": columns,
        "sample_count": int(len(rows)),
        "nominal_sample_rate_hz": FS_NOMINAL,
        "approx_duration_s": len(rows) / FS_NOMINAL,
        "counter_first_last": [int(counter[0]), int(counter[-1])],
        "counter_modulus": 256,
        "counter_nonunit_transitions": int(np.sum(transitions != 1)),
        "counter_wraps": int(np.sum(np.diff(counter) < 0)),
        "all_channel_values_finite": bool(np.isfinite(x).all()),
        "channel1_candidate_pulse_peaks": {
            "count": int(len(peaks)),
            "approx_times_s": [round(float(envelope_t[i]), 3) for i in peaks],
            "rule": "100 ms mean-centered RMS; threshold 0.04; minimum peak separation 0.30 s; prominence 0.018. Exploratory, not validated labels."
        },
        "lower_threshold_channelwise_candidate_peaks": {
            f"Channel{i+1}": {
                "count": int(len(per_channel_peaks[i][0])),
                "height_threshold": peak_thresholds[i],
                "minimum_peak_separation_s": 0.30,
            }
            for i in range(3)
        },
        "channel_statistics": stats,
        "method": "Read-only CSV profile. RMS uses mean-centered 100-sample windows with 50-sample hop. No filtering, segmentation, event labels, or model fitting.",
        "limitations": [
            "CSV has no per-sample timestamps; seconds are approximate and assume nominal 1000 samples/s.",
            "CSV has no cue/event labels, so the single/double/triple trial order and motion onsets cannot be established from metadata.",
            "Channels contain processed floating-point values, not untouched ADC counts; physical units are not present in this CSV."
        ]
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
