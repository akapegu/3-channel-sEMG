# Review of the archived Gemini code snapshots

## What is in the folder

The folder contains four plots and five code snippets about the four supplied swipe CSVs. The plots show per-file power spectra, full traces, a zoom/filter view for `swipe_down`, and high-pass energy envelopes for all four recordings. They are derived views of the same four files; they are not additional recordings.

## Useful observations

- The energy plot shows repeated bursts in all four takes. `swipe_left` has fairly regular repeated bursts; `swipe_down` has repeated Channel 2-dominant pulses with later Channel 3 activity; `swipe_top` and `swipe_right` look more variable.
- The full traces and per-channel views are useful exploratory visualizations. They support checking channel distribution and repetition timing before designing labels/windows.
- These observations are consistent with, but do not strengthen beyond, the separate EXP-013 exploratory within-take baseline (RBF-SVM 60.1% accuracy, macro-F1 0.597 with one file per class).

## Important limitations in the snippets

1. The CSVs have sample counters, not timestamps. Several snippets assume `fs=500 Hz` based on assumed recording duration / spectral interpretations. The counter cannot establish SPS by itself. The project has evidence for roughly 1000 SPS from other measurements; at a nominal 1000 SPS these files last about 14–20 seconds. Any Hz-labeled filter/spectrum result must state and verify the rate first.
2. `periodogram(..., fs=1.0)` and `rfftfreq(..., d=1.0)` report cycles per sample, not physical Hz. A normalized-frequency peak alone does not establish 50/60 Hz interference or the sample rate.
3. The Butterworth calls use `Wn=0.05`, a fraction of Nyquist. Their physical cutoff changes with SPS. The plot's label “true sEMG” is too strong: high-pass filtering can reduce slow movement artifact but does not prove the remainder is muscle-only.
4. `filtfilt` is zero-phase and uses future samples. It is suitable for offline visualization here, not a direct latency claim or a causal live-game detector.
5. The burst detector uses a baseline from the first 700 samples, a fixed relative height/prominence, and a fixed `distance=550`. The initial samples have not been independently confirmed as rest for every file. A minimum distance can merge close repetitions, and the resulting peaks are candidates, not validated labels.
6. No end-to-end classifier evaluation appears in these snippets. The feature-extraction file defines a large feature vector but does not build windows, fit a classifier, or evaluate held-out repetitions. The imports of CV and XGBoost do not constitute model results. Several files also rely on variables (`files`, `pd`, `plt`, `classes`, `dfs`, `signal`) created elsewhere, so they are notebook fragments rather than standalone scripts.

## Recommended use

Keep the raw/browser exports unchanged. First confirm the timing basis and mark the motion/release intervals from visible cues or reliable annotations. Use the Gemini plots as exploratory views, not labels. For performance claims, collect multiple separately cued takes per swipe and hold out complete repetitions or sessions. Fit feature scaling, thresholds, and models only on the training side; report confusion and latency on untouched repetitions. The feature-heavy proposal should be pared down to a small RMS/MAV/waveform-length baseline first, then add features or a temporal model only if held-out data show a gain.
