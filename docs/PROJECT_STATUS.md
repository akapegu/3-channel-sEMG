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
- Leave-one-session-out validation trained on two Sessions and scored the third. Best of five candidates was Extra Trees fusion at 400 ms: 305/360 event intervals (84.7%), 189/240 gestures (78.75%), and 4/120 tilt-only false commands. TOP was predicted REST on 11/60 trials in held-out Sessions 2–3. This is development validation and remains below target. A provisional model fit on all three sessions is saved locally without a validation claim.
- The frozen Session 1–3 model was scored once on Session 4: 68/120 events correct (56.7%), 46/80 gestures correct (57.5%), and 18/40 relaxed-finger tilt controls falsely called commands. Thus the 3-fold S1–S3 screen did not predict this wearing's result; Session 4 is consumed and cannot be retuned and reused as an independent test.
- Cue-level audit of held-out predictions found gesture accuracy of 66/80 (82.5%), 61/80 (76.3%), 62/80 (77.5%), and 46/80 (57.5%) on S1–S4, respectively. Tilt false commands were 4/40, 0/40, 0/40, and 18/40. S1–S3 gesture misses were mostly REST confusions; Session 4 shifted to RIGHT confusions, including 15/40 tilt controls predicted RIGHT.
- Session 4 relaxed-cue processed EMG RMS medians were CH1/CH2/CH3 23.0/4.0/25.4, compared with S1–S3 values 12.2–16.2 / 2.8–4.1 / 5.9–9.2. This is a measured signal-distribution change; its cause (placement/contact, execution, noise, or other variation) is unknown.
- The fixed Extra Trees model's impurity feature importance summed to 61.3% over EMG features and 38.7% over accelerometer features. This is an internal split-use summary, not causal evidence.
- A supplier-style CNN-LSTM was then compared across four complete held-out wearings. EMG-only reached 67.8% pooled gesture accuracy with 16/160 tilt controls falsely called commands; fused EMG+accelerometer reached 48.4% gesture accuracy with 2/160 false commands. These are development comparisons, not an independent final result; the four sessions were used for model/modality selection.
- Separate five-fold, within-wearing cue-grouped CNN-LSTM checks found fused gesture accuracy of 71.3%, 62.5%, 67.5%, and 62.5% for Sessions 1–4, with 4, 0, 3, and 1 false commands on 40 tilt controls respectively. EMG-only gesture accuracy was 41–56% and produced 14–35/40 false commands per session. Same-wearing performance is therefore still below the project's target; these checks do not replace whole-session evaluation.

## Assumptions

- Firmware's configured 1000 SPS/channel is the nominal rate used to map samples to seconds. Host-arrival rate is an estimate, not a calibrated sensor clock.
- Cue boundaries approximate movement timing; user reaction and movement onset occur after the cue and are not individually instrumented.

## Planned

- Use the cue and signal audit to plan one new development re-wear (Session 5). Keep gesture definitions and cue schedule unchanged; record approximate band location/orientation and fit/contact checks. Do not tune on Session 4 and describe it as independent validation.
- After Session 5 QC, compare complete-session held-out predictions and decide whether another development re-wear is needed. Freeze the candidate before collecting a new untouched final wearing.
- Measure false activations during ordinary no-command activity, per-gesture event misses, and movement-to-action latency; existing REST examples are prompted tilt controls, not daily activity.
- Evaluate compact hardware, embedded inference, and BLE HID only after laptop-side live detection has measured latency and false activations.

## References

- [Project-specific sEMG literature review](semg_literature_review.md)
- [Hudgins et al. time-domain myoelectric features](https://pubmed.ncbi.nlm.nih.gov/8468080/)
- [EMG window leakage study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9797837/)
- [scikit-learn GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [Window-duration evaluation for sEMG gesture classification](https://pmc.ncbi.nlm.nih.gov/articles/PMC6946760/)
- [EMGBench cross-distribution generalization benchmark](https://arxiv.org/abs/2410.23625)
