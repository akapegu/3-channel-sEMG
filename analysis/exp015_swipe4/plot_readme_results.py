"""Generate README figures from the documented four-session evaluation totals.

The figures contain aggregate evaluation metrics only; no biosignal traces,
cue-level values, raw packet data, or model weights are included.
"""
from pathlib import Path


OUT = Path(__file__).resolve().parent / "figures"
SESSIONS = ["S1", "S2", "S3", "S4"]
GESTURE_CORRECT = [66, 61, 62, 46]
GESTURE_TOTAL = [80, 80, 80, 80]
TILT_FALSE = [4, 0, 0, 18]
TILT_TOTAL = [40, 40, 40, 40]
BLUE = "#2B6F9C"
ORANGE = "#C65D21"
INK = "#20313D"
MUTED = "#5C6C76"
GRID = "#D8E0E5"


def esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def label(x: float, y: float, text: str, size: int = 14, anchor: str = "middle",
          color: str = INK, weight: str = "400", rotate: str = "") -> str:
    transform = f' transform="{rotate}"' if rotate else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{transform} '
            f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{color}">{esc(text)}</text>')


def session_chart() -> str:
    width, height = 1120, 570
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Four-session gesture recognition and tilt false-command results</title>',
           '<desc id="desc">Grouped panels show gesture-cue accuracy and the percentage of relaxed-finger tilt controls falsely classified as a command for four held-out wearings. Session 4 accuracy fell and false commands rose.</desc>',
           f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>',
           label(width / 2, 34, "Four-gesture performance changes after re-wearing", 23, weight="600"),
           label(width / 2, 61, "Fixed 400 ms Extra Trees + EMG/accelerometer features; each event is one prompted cue", 14, color=MUTED)]
    panels = [(55, "Gesture cues classified correctly", GESTURE_CORRECT, GESTURE_TOTAL, "correct"),
              (585, "Tilt-only controls falsely called commands", TILT_FALSE, TILT_TOTAL, "false")]
    plot_top, plot_bottom = 125, 410
    plot_h = plot_bottom - plot_top
    for px, title, nums, dens, kind in panels:
        plot_left, plot_right = px + 62, px + 475
        svg.append(label((plot_left + plot_right) / 2, 100, title, 17, weight="600"))
        for tick in range(0, 101, 20):
            y = plot_bottom - plot_h * tick / 100
            svg.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
            svg.append(label(plot_left - 12, y + 5, f"{tick}%", 12, anchor="end", color=MUTED))
        centers = [plot_left + 50 + i * 100 for i in range(4)]
        for i, center in enumerate(centers):
            pct = 100 * nums[i] / dens[i]
            bar_h = plot_h * pct / 100
            color = ORANGE if i == 3 else BLUE
            svg.append(f'<rect x="{center-25}" y="{plot_bottom-bar_h:.1f}" width="50" height="{bar_h:.1f}" rx="3" fill="{color}"/>')
            rounded_pct = int(pct * 10 + 0.5) / 10
            value = f"{nums[i]}/{dens[i]}  ({rounded_pct:.1f}%)" if kind == "correct" else f"{nums[i]}/{dens[i]}  ({pct:.0f}%)"
            svg.append(label(center, max(plot_top + 18, plot_bottom - bar_h - 10), value, 12, weight="600"))
            svg.append(label(center, plot_bottom + 24, SESSIONS[i], 14, weight="600"))
            svg.append(label(center, plot_bottom + 43, "locked" if i == 3 else "LOSO", 11, color=MUTED))
        svg.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="{INK}" stroke-width="1.2"/>')
    svg.append(label(width / 2, 496, "S1–S3: leave-one-session-out development folds (each model trained on the other two sessions).", 13, color=MUTED))
    svg.append(label(width / 2, 518, "S4: one locked evaluation using the model trained on S1–S3; this test is now consumed.", 13, color=MUTED))
    svg.append(label(width / 2, 540, "Each session: 80 gesture cues and 40 relaxed-finger tilt controls. Not a live-latency or daily false-command test.", 12, color=MUTED))
    svg.append('</svg>')
    return "\n".join(svg) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "four-session-results.svg").write_text(session_chart())
    print(f"Wrote README figures to {OUT}")


if __name__ == "__main__":
    main()
