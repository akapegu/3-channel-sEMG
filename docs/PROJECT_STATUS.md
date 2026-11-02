# Project status

## Measured

- Completed four-swipe BLE capture: 417,520 EMG samples, 96 cued movements, no EMG counter gaps, and an estimated host-arrival rate of 1000.024 frames/s. LIS3DH packets also had no observed counter discontinuities in that capture.
- Locked offline model: shrinkage LDA with EMG and accelerometer features; window features are calculated over 200 samples at a 20-sample hop. Same-session test: accuracy 0.797, balanced accuracy 0.673, macro-F1 0.688. Cued-event majority vote: 27/32 correct.
- The four-direction test split was a later chronological block of the same continuous wearing. It is not an independent re-wear/day evaluation.
- Live shadow mode has shown per-window predictions but had no movement labels during free-running captures. These predictions do not measure classifier accuracy.
- The separate clench game currently uses a 20 ms RMS window and 10 ms update hop. Its runtime event counts are not accuracy or response-latency results without movement timestamps.
- The first saved fast-detector game run contained 537,740 filtered samples over 537.78 s and 818 threshold-trigger events. There were no time-aligned clench annotations, so this does not show 818 intentional clenches or indicate accuracy. Raw and filtered source files remain local and are excluded from Git.

## Assumptions

- Firmware's configured 1000 SPS/channel is the nominal rate used to map samples to seconds. Host-arrival rate is an estimate, not a calibrated sensor clock.
- Cue boundaries approximate movement timing; user reaction and movement onset occur after the cue and are not individually instrumented.

## Planned

- Collect a separate re-wear session using the same four swipe definitions and evaluate the locked preprocessing/model before changing the model.
- Measure false activations during rest and ordinary activity, per-gesture event misses, and movement-to-action latency.
- Consider a separately trained longer-window model only after collecting comparable new data; do not change the inference window of the current trained model alone.
- Evaluate compact hardware and BLE HID only after laptop-side live behavior is measured and stable.

## References

- [Project-specific sEMG literature review](semg_literature_review.md)
- [Hudgins et al. time-domain myoelectric features](https://pubmed.ncbi.nlm.nih.gov/8468080/)
- [EMG window leakage study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9797837/)
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
