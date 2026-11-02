#!/usr/bin/env python3
"""Replay the inspected browser's Off/Off and Off/50Hz paths, not corrected filters.
Run from project root: scripts/.venv/bin/python scripts/compare_browser_processing.py
All filter states begin at zero at the first saved sample (an explicit assumption).
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/exp003_comparison'
RAW = ROOT / 'recordings/exp003_attempt03_raw.csv'


def section(values, a1, a2, b0, b1, b2):
    """Browser's direct-form-II recurrence, independently per channel."""
    result = np.empty_like(values, dtype=float)
    z1 = np.zeros(3)
    z2 = np.zeros(3)
    for i, sample in enumerate(values):
        x = sample - a1 * z1 - a2 * z2
        result[i] = b0 * x + b1 * z1 + b2 * z2
        z2, z1 = z1, x
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with RAW.open() as f:
        rows = list(csv.DictReader(f))
    raw = np.array([[int(r[f'ch{c}_adc']) for c in (1, 2, 3)] for r in rows])
    # class g, 500 SPS branch; class p with signal-type Off still scales.
    off = section(raw, -1.99822285, .99822443,
                  .99911182, -1.99822364, .99911182) * (2 / 4096)
    # class v, 500 SPS, notch ID 1 (UI label '50Hz'), two sections.
    notch = section(off, -1.56858163, .96424138,
                    .96508099, -1.56202714, .96508099)
    notch = section(notch, -1.61100358, .96592171, 1, -1.61854514, 1)
    names = ['sample_index', 'nominal_time_s'] + [f'{kind}_ch{c}'
             for kind in ('raw', 'browser_off', 'browser_50hz') for c in (1, 2, 3)]
    t = np.arange(len(raw)) / 1000  # nominal acquisition axis, not host timestamps
    with (OUT / 'same_samples_processed.csv').open('x', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for i in range(len(raw)):
            writer.writerow([i, t[i], *raw[i], *off[i], *notch[i]])
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False})
    for label, mask in [('overview', np.ones(len(t), dtype=bool)),
                        ('detail_10_to_12s', (t >= 10) & (t < 12))]:
        fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True, layout='constrained')
        for c in range(3):
            axes[c, 0].plot(t[mask], raw[mask, c], lw=.55, color='#2864a0')
            axes[c, 1].plot(t[mask], off[mask, c], lw=.6, alpha=.8,
                            label='Signal Off / notch Off', color='#2864a0')
            axes[c, 1].plot(t[mask], notch[mask, c], lw=.6, alpha=.8,
                            label='Signal Off / notch “50Hz”', color='#c56b16')
            axes[c, 0].set_ylabel(f'CH{c+1} received counts')
            axes[c, 1].set_ylabel(f'CH{c+1} browser units')
            for ax in axes[c]:
                ax.grid(alpha=.2)
        axes[0, 0].set_title('Original Python samples')
        axes[0, 1].set_title('Same samples through browser code (500 SPS coefficients)')
        axes[0, 1].legend(fontsize=8, loc='upper right')
        for ax in axes[-1]:
            ax.set_xlabel('Time from sample index (s; nominal 1000 SPS)')
        fig.suptitle('EXP-003: raw vs browser replay — zero initial filter state')
        fig.savefig(OUT / f'{label}.png', dpi=160)
        plt.close(fig)
    summary = {
        'source_csv_sha256': hashlib.sha256(RAW.read_bytes()).hexdigest(),
        'browser_bundle_sha256': hashlib.sha256((OUT / 'source/browser-page.js').read_bytes()).hexdigest(),
        'source_url': 'https://amanmahe.github.io/npg-armband-visualizer/_next/static/chunks/app/page-e0a654f27d2308bb.js',
        'samples': len(raw), 'nominal_plot_sps': 1000,
        'browser_filter_configuration_sps': 500,
        'assumptions': ['zero filter state at first saved sample',
                        'signal-type Off on all channels',
                        'notch Off and notch 50Hz shown as separate scenarios'],
        'notes': ['Offline replay, not a browser export or a new acquisition.',
                  'Browser signed int16 decoding agrees with unsigned decoding for observed 0..4095 values.',
                  '500 SPS filter coefficients replayed unchanged at approximately 1000 SPS.',
                  'Browser output is dimensionless; no electrode-voltage calibration.'],
    }
    (OUT / 'processing_metadata.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
