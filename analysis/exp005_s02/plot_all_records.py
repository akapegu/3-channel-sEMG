"""Plot the original device recordings without modifying them.

Outputs:
  all_s02_trials_overview.png: cue-aligned S02 EMG envelope montage.
  all_device_emg_recordings.pdf: detailed EMG pages for all original captures.
  all_accelerometer_recordings.pdf: motion pages for every capture with IMU data.

Dense raw traces are rendered as min/max envelopes for readability. The source
CSV/JSONL files remain the authoritative, untouched sample records.
"""
from __future__ import annotations

import csv
import json
import math
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "supplier_toolkit/adapted/training_data"
OUT = Path(__file__).resolve().parent
EMG_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
IMU_UUID = "5a153fa9-7be0-400c-8ef8-d84502b31c4d"
COLORS = ("#2463a6", "#df7b19", "#269064")


def read_json(path: Path):
    return json.loads(path.read_text())


def decode_packets(path: Path, uuid: str, fmt: str):
    frames = []
    with path.open() as f:
        for line in f:
            packet = json.loads(line)
            if packet.get("characteristic") != uuid:
                continue
            payload = bytes.fromhex(packet["payload_hex"])
            if payload and len(payload) % 7 == 0:
                frames.extend(struct.iter_unpack(fmt, payload))
    return np.asarray(frames, dtype=np.int32)


def read_csv_columns(path: Path, wanted: tuple[str, ...]):
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        names = rows.fieldnames or []
        columns = {name: [] for name in wanted if name in names}
        for row in rows:
            for name in columns:
                columns[name].append(float(row[name]))
    return {k: np.asarray(v, dtype=np.float64) for k, v in columns.items()}


def cue_segments(csv_path: Path):
    sidecar = csv_path.with_suffix(".beep.json")
    if not sidecar.exists():
        return []
    doc = read_json(sidecar)
    return [(int(a), int(b)) for a, b in doc.get("segments", [])]


def bounded_envelope(values: np.ndarray, samples_per_bin: int):
    """Return bin centers and min/max envelopes, preserving narrow peaks."""
    n = len(values)
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    bins = max(1, math.ceil(n / samples_per_bin))
    lo = np.empty(bins, dtype=np.float64)
    hi = np.empty(bins, dtype=np.float64)
    centers = np.empty(bins, dtype=np.float64)
    for k, start in enumerate(range(0, n, samples_per_bin)):
        stop = min(n, start + samples_per_bin)
        block = values[start:stop]
        lo[k], hi[k] = np.min(block), np.max(block)
        centers[k] = (start + stop - 1) / 2
    return centers, lo, hi


def emg_record(path: Path, label: str, source: str, status: str, segments=None,
               rate=1000.0):
    """Read original ADC packet bytes when available, otherwise source CSV."""
    if segments is None:
        segments = []
    processing = path.with_suffix(".processing.json")
    raw_path = None
    if processing.exists():
        info = read_json(processing)
        candidates = info.get("raw_packet_files", [])
        if candidates:
            raw_path = Path(candidates[0])
    if raw_path and raw_path.exists():
        frames = decode_packets(raw_path, EMG_UUID, ">BHHH")
        emg = frames[:, 1:4].astype(np.float64)
        unit = "raw ADC counts"
        source += " · original BLE packets"
    else:
        cols = read_csv_columns(path, ("ch1_adc", "ch2_adc", "ch3_adc",
                                       "Channel1", "Channel2", "Channel3"))
        keys = next((names for names in (("ch1_adc", "ch2_adc", "ch3_adc"),
                                         ("Channel1", "Channel2", "Channel3"))
                     if all(name in cols for name in names)), None)
        if not keys:
            raise ValueError(f"No three EMG channels in {path}")
        emg = np.column_stack([cols[name] for name in keys])
        unit = "processed export units (not ADC counts)" if keys[0] == "Channel1" else "raw ADC counts"
        source += " · processed browser export" if keys[0] == "Channel1" else " · original CSV"
    return {"path": path, "label": label, "source": source, "status": status,
            "emg": emg, "unit": unit, "rate": rate, "segments": segments}


def load_s02_records():
    index = read_json(DATA / "dataset_index.json")
    split = read_json(DATA / "split_manifest.json")
    split_by_path = {item["path"]: item for item in split.get("files", [])}
    records = []
    subject = index["subjects"]["abinash_s02"]
    for gesture in ("rest", "pinch", "flexion", "extension"):
        for row in subject[gesture]["sessions"]:
            path = DATA / "abinash_s02" / gesture / row["csv"]
            assignment = split_by_path.get(path.relative_to(DATA).as_posix(), {})
            partition = assignment.get("partition", row.get("partition", "unassigned"))
            group = assignment.get("acquisition_group", row.get("acquisition_group", "unassigned"))
            if row.get("include_in_training"):
                status = f"{partition} · accepted"
            else:
                status = "excluded · redo" if "redo" in row.get("exclusion_reason", "").lower() else "pending"
            label = f"S02 · {gesture} trial {row['trial']} · {partition}"
            records.append(emg_record(path, label, group, status, cue_segments(path)))
    return records


def load_other_emg_records():
    records = []
    old = ROOT / "recordings"
    for path in sorted(old.glob("*.csv")):
        if path.name.endswith(".imu.csv"):
            continue
        records.append(emg_record(path, path.stem, "Earlier capture", "exploratory", []))
    live = ROOT / "supplier_toolkit/verification/livecheck03/raw_packets"
    for path in sorted(live.glob("*.jsonl")):
        frames = decode_packets(path, EMG_UUID, ">BHHH")
        records.append({"path": path, "label": "livecheck03 · diagnostic", "source": "raw BLE packets",
                        "status": "unlabelled diagnostic", "emg": frames[:, 1:4].astype(np.float64),
                        "unit": "raw ADC counts", "rate": 1000.0, "segments": []})
    return records


def load_accel_records(s02_records):
    records = []
    for record in s02_records:
        info = read_json(record["path"].with_suffix(".processing.json"))
        packets = info.get("raw_packet_files", [])
        if packets and Path(packets[0]).exists():
            frames = decode_packets(Path(packets[0]), IMU_UUID, ">Bhhh")
            records.append({"label": record["label"], "source": "original BLE packets",
                            "status": record["status"], "accel": frames[:, 1:4]})
    old = ROOT / "recordings"
    for path in sorted(old.glob("*.imu.csv")):
        cols = read_csv_columns(path, ("accel_x_counts", "accel_y_counts", "accel_z_counts"))
        names = ("accel_x_counts", "accel_y_counts", "accel_z_counts")
        if all(name in cols for name in names):
            records.append({"label": path.stem, "source": "original decoded IMU CSV",
                            "status": "earlier capture",
                            "accel": np.column_stack([cols[name] for name in names])})
    live = ROOT / "supplier_toolkit/verification/livecheck03/raw_packets"
    for path in sorted(live.glob("*.jsonl")):
        frames = decode_packets(path, IMU_UUID, ">Bhhh")
        records.append({"label": "livecheck03 · diagnostic", "source": "original BLE packets",
                        "status": "unlabelled diagnostic", "accel": frames[:, 1:4]})
    return records


def render_s02_overview(records, output):
    fig, axes = plt.subplots(4, 4, figsize=(18, 11), sharex=False, sharey=False)
    fig.suptitle("S02 gesture recordings · all trials", fontsize=17, y=0.978)
    fig.text(0.5, 0.944,
             "20 ms RMS envelope from raw ADC counts, centered and scaled within each channel/session; shaded = cue-labeled segment",
             ha="center", fontsize=10)
    colors = COLORS
    order = ("rest", "pinch", "flexion", "extension")
    for row_i, gesture in enumerate(order):
        group = [r for r in records if f"· {gesture} trial" in r["label"]]
        group.sort(key=lambda r: int(r["label"].split("trial ")[1].split(" ")[0]))
        for col_i in range(4):
            ax = axes[row_i, col_i]
            if col_i >= len(group):
                ax.axis("off")
                continue
            rec = group[col_i]
            x = np.arange(len(rec["emg"]))
            for a, b in rec["segments"]:
                ax.axvspan(a / rec["rate"], b / rec["rate"], color="#8abf9f", alpha=0.22, lw=0)
            for ch in range(3):
                raw = rec["emg"][:, ch]
                center = np.median(raw)
                centered = raw - center
                binsize = max(1, int(round(rec["rate"] * 0.02)))
                centers, _, _ = bounded_envelope(centered, binsize)
                rms = np.empty(len(centers), dtype=float)
                for k, start in enumerate(range(0, len(raw), binsize)):
                    z = centered[start:min(len(raw), start + binsize)]
                    rms[k] = np.sqrt(np.mean(z * z))
                scale = np.percentile(rms, 95) if len(rms) else 1.0
                ax.plot(centers / rec["rate"], rms / (scale if scale > 0 else 1),
                        color=colors[ch], lw=0.85, alpha=0.9, label=f"CH{ch+1}")
            ax.set_title(f"{rec['label'].split(' · ')[1]}\n{rec['status']}", fontsize=9)
            ax.set_xlim(0, len(rec["emg"]) / rec["rate"])
            ax.grid(alpha=0.18, linewidth=0.5)
            if col_i == 0:
                ax.set_ylabel(f"{gesture}\nrelative RMS")
            if row_i == 3:
                ax.set_xlabel("elapsed time (s; ~1 kSPS)")
            if rec["status"].startswith("excluded"):
                ax.patch.set_hatch("//")
    handles = [plt.Line2D([0], [0], color=COLORS[i], lw=1.2, label=f"CH{i+1}") for i in range(3)]
    handles.append(plt.Line2D([0], [0], color="#8abf9f", lw=5, alpha=0.5, label="labeled cue segment"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.text(0.5, 0.917,
             "Channels are normalized separately per recording for shape comparison; raw counts are in the detailed PDF.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.88))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_emg_pdf(records, pdf):
    with PdfPages(pdf) as out:
        per_page = 4
        for page_start in range(0, len(records), per_page):
            subset = records[page_start:page_start + per_page]
            fig, axes = plt.subplots(len(subset), 4, figsize=(16, max(7, 2.45 * len(subset))), squeeze=False,
                                     gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
            fig.suptitle(f"Original EMG captures · records {page_start+1}–{page_start+len(subset)} of {len(records)}",
                         fontsize=15, y=0.972)
            fig.text(0.5, 0.938, "Per-channel min/max envelope in 10 ms bins; cue spans are approximate sample-index timing. Raw source files are unchanged.",
                     ha="center", fontsize=9)
            for row, rec in enumerate(subset):
                n = len(rec["emg"])
                label_ax = axes[row, 0]
                label_ax.axis("off")
                label_ax.text(1.0, 0.5,
                              f"{rec['label']}\n{rec['status']}\n{rec['source']}",
                              ha="right", va="center", fontsize=8,
                              transform=label_ax.transAxes)
                for ch, ax in enumerate(axes[row, 1:]):
                    vals = rec["emg"][:, ch]
                    centers, lo, hi = bounded_envelope(vals, max(1, int(round(rec["rate"] * 0.01))))
                    ax.fill_between(centers / rec["rate"], lo, hi, color=COLORS[ch], alpha=0.32, linewidth=0)
                    ax.plot(centers / rec["rate"], (lo + hi) / 2, color=COLORS[ch], lw=0.55)
                    for a, b in rec["segments"]:
                        ax.axvspan(a / rec["rate"], b / rec["rate"], color="#8abf9f", alpha=0.16, lw=0)
                    ax.set_xlim(0, n / rec["rate"])
                    ax.grid(alpha=0.18, linewidth=0.5)
                    if row == 0:
                        ax.set_title(f"CH{ch+1} · {rec['unit']}")
                    if ch == 0:
                        ax.set_ylabel(rec["unit"], fontsize=8)
                    if row == len(subset) - 1:
                        ax.set_xlabel("sample-index time (s; nominal ~1 kSPS)")
            fig.tight_layout(rect=(0.025, 0.025, 0.99, 0.90))
            out.savefig(fig, dpi=150)
            plt.close(fig)


def render_accel_pdf(records, pdf):
    with PdfPages(pdf) as out:
        per_page = 6
        for page_start in range(0, len(records), per_page):
            subset = records[page_start:page_start + per_page]
            fig, axes = plt.subplots(len(subset), 2, figsize=(14, max(7, 1.7 * len(subset))), squeeze=False,
                                     gridspec_kw={"width_ratios": [1.5, 8]})
            fig.suptitle(f"Accelerometer captures · records {page_start+1}–{page_start+len(subset)} of {len(records)}",
                         fontsize=15, y=0.972)
            fig.text(0.5, 0.938, "Signed raw LIS3DH counts by motion-frame index; each session has its own frame counter origin; not synchronized to EMG sample time here.",
                     ha="center", fontsize=9)
            for row, rec in enumerate(subset):
                label_ax = axes[row, 0]
                label_ax.axis("off")
                label_ax.text(1.0, 0.5, f"{rec['label']}\n{rec['status']}",
                              ha="right", va="center", fontsize=8,
                              transform=label_ax.transAxes)
                ax = axes[row, 1]
                data = rec["accel"]
                for ch in range(min(3, data.shape[1])):
                    centers, lo, hi = bounded_envelope(data[:, ch], 1)
                    ax.plot(centers, data[:, ch], color=COLORS[ch], lw=0.55, label=f"axis {('X','Y','Z')[ch]}")
                ax.set_ylabel("counts", fontsize=8)
                ax.grid(alpha=0.18, linewidth=0.5)
                ax.set_xlim(0, len(data))
                if row == 0:
                    ax.legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
                if row == len(subset) - 1:
                    ax.set_xlabel("accelerometer frame index (~104 frames/s, measured receive rate)")
            fig.tight_layout(rect=(0.025, 0.025, 0.99, 0.90))
            out.savefig(fig, dpi=150)
            plt.close(fig)


def main():
    s02 = load_s02_records()
    other = load_other_emg_records()
    emg = s02 + other
    accel = load_accel_records(s02)
    overview = OUT / "all_s02_trials_overview.png"
    details = OUT / "all_device_emg_recordings.pdf"
    motion = OUT / "all_accelerometer_recordings.pdf"
    render_s02_overview(s02, overview)
    render_emg_pdf(emg, details)
    render_accel_pdf(accel, motion)
    print(f"S02 trials: {len(s02)}")
    print(f"All original EMG captures: {len(emg)}")
    print(f"Accelerometer captures: {len(accel)}")
    print(f"Overview: {overview}")
    print(f"Full EMG PDF: {details}")
    print(f"Full accelerometer PDF: {motion}")


if __name__ == "__main__":
    main()
