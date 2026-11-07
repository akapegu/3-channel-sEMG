"""Create README plots from the processed Session 4 three-channel EMG.

The CSV is a processed/filter-output recording, not raw ADC counts or calibrated
microvolts. Figures contain (1) representative cue waveforms selected by median
log-RMS proximity and (2) median/IQR RMS envelopes across every cue per class.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "supplier_toolkit/adapted/training_data/subject_01/session4/continuous/swipe4_control_trial1_20261104-003622_merged_ECF6.csv"
CUES = ROOT / "raw_packets/1793732227632908000_dev1.session_cues.json"
OUT = Path(__file__).resolve().parent / "figures"
CHANNELS = ["ch1", "ch2", "ch3"]
CLASSES = ["rest", "down", "left", "right", "top"]
CLASS_LABELS = {
    "rest": "REST (tilt-only control)", "down": "DOWN", "left": "LEFT",
    "right": "RIGHT", "top": "TOP",
}
COLORS = {
    "rest": "#586B75", "down": "#D89000", "left": "#0072B2",
    "right": "#008A66", "top": "#B35C92",
}
INK = "#20313D"
MUTED = "#5C6C76"
GRID = "#D8E0E5"
N_BINS = 30  # approximately 50 ms cue-relative bins at 1 kHz


def esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x: float, y: float, value: str, size=13, anchor="middle", color=INK, weight="400", transform="") -> str:
    tr = f' transform="{transform}"' if transform else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{tr} '
            f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{esc(value)}</text>')


def action_intervals(events: list[dict], count: int) -> list[dict]:
    starts = {}
    out = []
    for event in events:
        if event.get("phase") != "ACTION" or event.get("event") not in {"label_start", "label_stop"}:
            continue
        phase_index = int(event["phase_index"])
        if event["event"] == "label_start":
            starts[phase_index] = event
            continue
        begin = starts.pop(phase_index)
        start, stop = int(begin["sample_index"]), int(event["sample_index"])
        if not (0 <= start < stop <= count):
            raise ValueError(f"Invalid cue indices [{start}, {stop})")
        out.append({"start": start, "stop": stop, "label": begin["label"],
                    "trial_type": begin.get("trial_type"),
                    "direction": begin.get("direction"),
                    "trial_index": int(begin["trial_index"])})
    if starts:
        raise ValueError("Unclosed action label interval")
    return sorted(out, key=lambda r: r["start"])


def read_signals() -> tuple[list[list[float]], list[dict]]:
    cue = json.loads(CUES.read_text())
    if cue.get("status") != "complete" or cue.get("session") != "session4":
        raise ValueError("Expected the complete Session 4 cue timeline")
    data = [[] for _ in CHANNELS]
    with DATA.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for channel, name in enumerate(CHANNELS):
                data[channel].append(float(row[name]))
    intervals = action_intervals(cue["events"], len(data[0]))
    if len(intervals) != 120:
        raise ValueError(f"Expected 120 action intervals, got {len(intervals)}")
    return data, intervals


def cue_rms(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))


def qtile(values: list[float], p: float) -> float:
    s = sorted(values)
    position = (len(s) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (position - lo)


def choose_representatives(data: list[list[float]], intervals: list[dict]) -> dict[str, dict]:
    candidates = {cls: [] for cls in CLASSES}
    for interval in intervals:
        cls = interval["label"]
        if cls not in candidates:
            raise ValueError(f"Unexpected class {cls}")
        rms = [cue_rms(signal[interval["start"]:interval["stop"]]) for signal in data]
        candidates[cls].append((interval, rms))
    chosen = {}
    for cls, rows in candidates.items():
        if not rows:
            raise ValueError(f"Missing cues for {cls}")
        med = [statistics.median(row[1][ch] for row in rows) for ch in range(3)]
        scale = [max(statistics.median([abs(row[1][ch] - med[ch]) for row in rows]), 1e-6)
                 for ch in range(3)]
        interval, _ = min(rows, key=lambda row: sum(
            ((math.log1p(row[1][ch]) - math.log1p(med[ch])) / max(scale[ch] / (med[ch] + 1), 1e-3)) ** 2
            for ch in range(3)))
        chosen[cls] = interval
    return chosen


def waveform_svg(data: list[list[float]], chosen: dict[str, dict]) -> str:
    width, height = 1180, 840
    left, right = 105, 1140
    panel_top, panel_h, gap = 135, 195, 38
    gap_samples = 250
    sequence = []
    offset = 0
    for cls in CLASSES:
        interval = chosen[cls]
        start, stop = interval["start"], interval["stop"]
        length = stop - start
        sequence.append((cls, offset, offset + length, start, stop))
        offset += length + gap_samples
    total_samples = offset - gap_samples
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Three processed EMG channel waveforms during representative Session 4 cues</title>',
           '<desc id="desc">Three stacked time-series panels show one typical 1.5 second cue for REST tilt control and each of DOWN, LEFT, RIGHT, TOP. A median log-RMS cue was selected for each class. The 250 millisecond gaps between cue snippets are artificial.</desc>',
           f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
           txt(width/2, 34, "Three-channel EMG during representative gesture cues", 22, weight="600"),
           txt(width/2, 61, "Session 4 processed filter output; one median-RMS example per label", 14, color=MUTED),
           txt(width/2, 83, "Cues are approximately 1.5 s; 250 ms display gaps are inserted between excerpts", 13, color=MUTED)]
    plot_w = right-left
    for ch, signal in enumerate(data):
        ytop = panel_top + ch*(panel_h+gap)
        ybottom = ytop + panel_h
        chunks = []
        max_abs = 0.0
        for cls, disp_start, disp_stop, source_start, source_stop in sequence:
            vals = signal[source_start:source_stop]
            chunks.append((cls, disp_start, vals))
            max_abs = max(max_abs, max(abs(v) for v in vals))
        ylim = max(max_abs*1.1, 1.0)
        for tick in (-1, -0.5, 0, 0.5, 1):
            y = (ytop+ybottom)/2 - tick*(panel_h*0.43)
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            svg.append(txt(left-12, y+4, f"{tick*ylim:.0f}", 11, anchor="end", color=MUTED))
        for cls, disp_start, disp_stop, source_start, source_stop in sequence:
            x1 = left + plot_w*disp_start/total_samples
            x2 = left + plot_w*disp_stop/total_samples
            svg.append(f'<rect x="{x1:.1f}" y="{ytop}" width="{x2-x1:.1f}" height="{panel_h}" fill="{COLORS[cls]}" fill-opacity="0.055"/>')
            duration = (source_stop-source_start)/1000
            svg.append(txt((x1+x2)/2, ytop-9, f"{CLASS_LABELS[cls]} ({duration:.2f}s)", 12, color=COLORS[cls], weight="600"))
        # About one plotted point every 2 ms keeps SVG compact while retaining waveform shape.
        for cls, disp_start, vals in chunks:
            segment = []
            for local in range(0, len(vals), 2):
                x = left + plot_w*(disp_start+local)/total_samples
                y = (ytop+ybottom)/2 - (vals[local]/ylim)*(panel_h*0.43)
                segment.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{INK}" stroke-width="1.15" stroke-linejoin="round" stroke-linecap="round"/>')
        svg.append(txt(22, (ytop+ybottom)/2, f"{CHANNELS[ch].upper()} processed value", 13, color=INK, weight="600", transform=f"rotate(-90 22 {(ytop+ybottom)/2})"))
        svg.append(f'<line x1="{left}" y1="{ybottom}" x2="{right}" y2="{ybottom}" stroke="{INK}" stroke-width="1"/>')
    svg.append(txt(width/2, 824, "Separate cue excerpts; the shaded intervals are not one continuous recording timeline", 13, color=MUTED))
    svg.append('</svg>')
    return "\n".join(svg)+"\n"


def envelope_svg(data: list[list[float]], intervals: list[dict]) -> str:
    width, height = 1180, 875
    left, right = 105, 1140
    panel_top, panel_h, gap = 145, 190, 45
    bins_per_cue = N_BINS
    stats = {}
    for cls in CLASSES:
        events = [iv for iv in intervals if iv["label"] == cls]
        if not events:
            raise ValueError(f"No cues for {cls}")
        stats[cls] = []
        for ch, signal in enumerate(data):
            values_by_bin = [[] for _ in range(bins_per_cue)]
            for iv in events:
                for b in range(bins_per_cue):
                    duration = iv["stop"] - iv["start"]
                    lo = iv["start"] + math.floor(b*duration/bins_per_cue)
                    hi = iv["start"] + math.floor((b+1)*duration/bins_per_cue)
                    seg = signal[lo:hi]
                    values_by_bin[b].append(math.sqrt(sum(x*x for x in seg)/len(seg)))
            stats[cls].append([(statistics.median(vals), qtile(vals,.25), qtile(vals,.75)) for vals in values_by_bin])
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Three-channel EMG RMS envelopes by cued class</title>',
           '<desc id="desc">Three panels show median and interquartile range across all Session 4 repetitions for four gestures and relaxed-finger tilt controls. Each value is RMS within a 50 millisecond bin of the processed EMG channel.</desc>',
           f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
           txt(width/2, 34, "Channel activity profiles across all prompted cues", 22, weight="600"),
           txt(width/2, 61, "Session 4 processed EMG; 30 cue-relative RMS bins; median line and middle-50% band", 14, color=MUTED),
           txt(width/2, 80, "20 repetitions per gesture; 40 relaxed-finger tilt-control cues", 13, color=MUTED)]
    legend_x = 188
    for i, cls in enumerate(CLASSES):
        x = legend_x + i*202
        svg.append(f'<line x1="{x}" y1="111" x2="{x+28}" y2="111" stroke="{COLORS[cls]}" stroke-width="3"/>')
        svg.append(txt(x+36, 116, CLASS_LABELS[cls], 12, anchor="start", color=INK))
    plot_w = right-left
    for ch in range(3):
        ytop = panel_top + ch*(panel_h+gap)
        ybottom = ytop + panel_h
        max_v = max(stats[cls][ch][b][2] for cls in CLASSES for b in range(bins_per_cue))*1.08
        max_v = max(max_v, 1.0)
        for tick in range(5):
            value = max_v*tick/4
            y = ybottom - panel_h*tick/4
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            svg.append(txt(left-12, y+4, f"{value:.0f}", 11, anchor="end", color=MUTED))
        svg.append(txt(22, (ytop+ybottom)/2, f"{CHANNELS[ch].upper()} RMS", 13, color=INK, weight="600", transform=f"rotate(-90 22 {(ytop+ybottom)/2})"))
        for tick_ms in (0, 250, 500, 750, 1000, 1250, 1500):
            x = left + plot_w*tick_ms/1500
            svg.append(f'<line x1="{x:.1f}" y1="{ybottom}" x2="{x:.1f}" y2="{ybottom+4}" stroke="{INK}"/>')
            svg.append(txt(x, ybottom+17, str(tick_ms), 10, color=MUTED))
        for cls in CLASSES:
            rows = stats[cls][ch]
            upper = []
            lower = []
            median = []
            for b, (med, q1, q3) in enumerate(rows):
                x = left + plot_w*(b+.5)/bins_per_cue
                upper.append((x, ybottom-panel_h*q3/max_v))
                lower.append((x, ybottom-panel_h*q1/max_v))
                median.append((x, ybottom-panel_h*med/max_v))
            poly = upper + list(reversed(lower))
            points = " ".join(f"{x:.1f},{y:.1f}" for x,y in poly)
            svg.append(f'<polygon points="{points}" fill="{COLORS[cls]}" fill-opacity="0.12" stroke="none"/>')
            line_points = " ".join(f"{x:.1f},{y:.1f}" for x,y in median)
            svg.append(f'<polyline points="{line_points}" fill="none" stroke="{COLORS[cls]}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>')
        svg.append(f'<line x1="{left}" y1="{ybottom}" x2="{right}" y2="{ybottom}" stroke="{INK}" stroke-width="1"/>')
    svg.append(txt(width/2, 862, "Cue-relative time (ms; each cue normalized to 1.5 s)", 13, color=MUTED))
    svg.append('</svg>')
    return "\n".join(svg)+"\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    signals, intervals = read_signals()
    representative = choose_representatives(signals, intervals)
    (OUT / "session4-three-channel-waveforms.svg").write_text(waveform_svg(signals, representative))
    (OUT / "session4-three-channel-envelopes.svg").write_text(envelope_svg(signals, intervals))
    print(f"Wrote two three-channel Session 4 plots to {OUT}")


if __name__ == "__main__":
    main()
