# Project status

## Measured

- Completed four-swipe BLE capture: 417,520 EMG samples, 96 cued movements, no EMG counter gaps, and an estimated host-arrival rate of 1000.024 frames/s. LIS3DH packets also had no observed counter discontinuities in that capture.
- Locked offline model: shrinkage LDA with EMG and accelerometer features; window features are calculated over 200 samples at a 20-sample hop. Same-session test: accuracy 0.797, balanced accuracy 0.673, macro-F1 0.688. Cued-event majority vote: 27/32 correct.
- The four-direction test split was a later chronological block of the same continuous wearing. It is not an independent re-wear/day evaluation.
- Live shadow mode has shown per-window predictions but had no movement labels during free-running captures. These predictions do not measure classifier accuracy.
- The separate clench game currently uses a 20 ms RMS window and 10 ms update hop. Its runtime event counts are not accuracy or response-latency results without movement timestamps.
- The first saved fast-detector game run contained 537,740 filtered samples over 537.78 s and 818 threshold-trigger events. There were no time-aligned clench annotations, so this does not show 818 intentional clenches or indicate accuracy. Raw and filtered source files remain local and are excluded from Git.
- A separate four-gesture re-wear control session recorded 512,620 samples with continuous EMG and accelerometer counters: 20 examples per gesture plus 10 relaxed-finger tilt controls per direction. The participant confirmed following all cues. The raw recording, packet bytes, cue timeline, and participant-specific metadata remain local and are excluded from Git.
- In a same-session grouped screen, Extra Trees + EMG/accelerometer features with 400 ms windows classified 68/80 gesture intervals (85%) and falsely called 2/40 tilt-only controls commands. LEFT was 12/20, so this is not yet a reliable four-gesture baseline. The screen compared 100–400 ms windows and several models on the same folds; its highest score is optimistic and not an independent test.
- A second capture after re-wearing the band recorded 512,580 samples. EMG and accelerometer counters were continuous, processed values were finite, and all 120 cues were complete and in range. A candidate trained only on Session 1—Extra Trees with EMG+accelerometer features and a 400 ms window—scored 68/80 gestures (85%) and made 1 false command on 40 relaxed-finger tilt controls in Session 2. RIGHT was 14/20. This same-day cross-placement result is still development evidence; Session 2 was used to screen candidates, and latency was not measured.
- Session 3 also passed capture QC (512,580 samples; continuous counters; 120 complete cues). The fixed 400 ms fused Extra Trees candidate trained on Sessions 1+2 scored 62/80 gestures (77.5%) and made 0/40 tilt-control false commands. TOP recall was 13/20, all predicted REST. This does not meet the 90% target and does not measure ordinary activity or latency.

## Assumptions

- Firmware's configured 1000 SPS/channel is the nominal rate used to map samples to seconds. Host-arrival rate is an estimate, not a calibrated sensor clock.
- Cue boundaries approximate movement timing; user reaction and movement onset occur after the cue and are not individually instrumented.

## Planned

- Evaluate the 300 and 400 ms candidates with sequential replay and independently marked movement onset to measure detection delay. Keep the currently locked 200 ms model unchanged unless a new model is trained and validated with matching features.
- Use Sessions 1–3 for session-held-out candidate validation and targeted diagnosis of TOP→REST errors. Freeze the model/window pipeline before evaluating once on the untouched Session 4.
- Measure false activations during rest and ordinary activity, per-gesture event misses, and movement-to-action latency.
- Evaluate compact hardware and BLE HID only after laptop-side live behavior is measured and stable.

## References

- [Project-specific sEMG literature review](semg_literature_review.md)
- [Hudgins et al. time-domain myoelectric features](https://pubmed.ncbi.nlm.nih.gov/8468080/)
- [EMG window leakage study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9797837/)
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Window-duration evaluation for sEMG gesture classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC6946760/)
- [EMGBench cross-distribution generalization benchmark](https://arxiv.org/abs/2410.23625)
