# Four-swipe baseline, EXP-015/016

## Status

One complete, user-confirmed continuous capture has been segmented into `rest`, `down`, `left`, `right`, and `top`. The original source CSV, packet-byte JSONL, and visual-cue event file are preserved. A regularized LDA+accelerometer-fusion baseline has been fit on the train and validation partitions and evaluated once on the held-out chronological test block.

## Data flow

1. `supplier_toolkit/adapted/record_swipe4_session.py` scans for one NPG-Lite 27:BC band, waits for a visual Start click, then opens one BLE connection and starts one uninterrupted stream. Exact EMG/IMU notification bytes go to JSONL; the existing `DeviceRecorder` outputs filtered three-channel EMG, sample-held LIS3DH axes, and byte counters in the continuous CSV.
2. `verify_guided_session.py` checks raw packet shape, frame counts, EMG and IMU counter continuity, host-arrival rate, processed row count, finite values, and cue sample-index bounds. Host-arrival rate is an estimate; it is not calibrated sensor-clock timing.
3. The user confirmed following the full cue sequence. `segment_swipe4_session.py` pairs cue `label_start`/`label_stop` records by phase index, checks every sample range and repetition count, and writes five labels into each chronological partition. Each sidecar preserves source ranges and local segment ranges. No window crosses a cue interval or concatenated-repetition boundary.
4. `train_swipe4_baseline.py --stage validate` opens only train and validation data. For each cue segment it extracts 200-sample windows with a 20-sample hop. EMG features are channel-wise log RMS, MAV, and mean absolute waveform length. Fusion adds the mean and standard deviation for each accelerometer axis. `StandardScaler` is inside each scikit-learn pipeline, so it is fit on the model's fitting rows only.
5. Four candidates were compared on validation only: shrinkage LDA and RBF SVM, each with EMG-only or EMG+accelerometer features. The highest validation window macro-F1 selected shrinkage LDA+fusion. `--stage lock` refit that fixed choice on train+validation; the saved JSON records that the test files were not read during selection.
6. `--stage test` then opened the test partition once and saved the locked result. Do not use this consumed block for further model selection. A new re-wear recording is needed for an independent placement/session check.

## Results

- Validation-selected candidate: LDA+fusion, validation window macro-F1 0.658; active-cue majority vote 32/32.
- Locked test: window accuracy 0.797, balanced accuracy 0.673, macro-F1 0.688.
- Test majority vote across 32 prompted swipe intervals: 27/32 (down 7/8, left 6/8, right 8/8, top 6/8). Five gesture intervals were voted as REST.
- These are within-session offline results. Overlapping windows are correlated and not independent repetitions; test is from the same wearing/session. This does not establish live command success, natural-activity false-command rate, or re-wear/day generalization.

## Artifacts

- Source QC: `trial3_qc.json`
- Cue-bounded split manifest: `swipe4_split_manifest_v2.json`
- Validation comparison (test unopened): `model_v1/validation_selection.json`
- Frozen LDA model and configuration: `model_v1/locked_model.joblib`, `model_v1/locked_model.json`
- Single locked test evaluation: `model_v1/locked_test_metrics.json`
- Confusion plot: `model_v1/locked_test_confusion.png`
- Capture and model history: `experiment_list.md`

## Re-wear robustness follow-up

The follow-up protocol explicitly separates the four intended finger actions from relaxed-finger tilt-only controls. It uses one continuous capture per re-wear session, compares EMG-only / accelerometer-only / fusion, keeps complete repetitions together during development, and reserves the fourth session for a locked test. See [`robust_protocol.md`](robust_protocol.md) and [`robust_session_template.json`](robust_session_template.json). Session 1 is QC-passed: 20 repetitions per gesture and 10 tilt-only controls per direction, 512,620 samples, continuous counters, and all 120 action cues complete. The user confirmed cue compliance. Source recordings and identifying capture metadata remain local and are not tracked.

### Session 1 accuracy checks (development only)

- The old frozen `LDA_FUSION` model scored 64/120 action cues (53.3%) by event vote on Session 1; it recognized 27/80 intended finger gestures (33.8%) and falsely called 3/40 tilt-only controls commands. This evaluates transfer from the older recording; it did not train on Session 1.
- Five-fold stratified group CV, holding out whole action repetitions, gave `SVM_FUSION` 90/120 (75.0%) event accuracy and 62/80 (77.5%) gesture accuracy, but it falsely called 12/40 tilt-only controls commands. `SVM_EMG` scored 81/120 (67.5%) and 44/80 gestures (55.0%), with 3/40 tilt-only false commands. Accelerometer-only SVM scored 44/120 (36.7%) and falsely called 32/40 tilt-only controls commands.
- These are different evaluation questions: the first is the old model on a new recording; the second trains and validates within this single wearing. The within-session result is a development estimate, not re-wear performance. LEFT remains weak (9/20 recall for the best within-session candidate), so the baseline is not ready for control. Session 1 is now development data; Session 4 remains untouched for the final evaluation. No HID action was enabled.

### Richer model screen (2026-11-03; exploratory)

At the user's request, compared Random Forest, Extra Trees, and a small two-layer MLP against the existing RBF SVM using the same Session 1 features, event-vote metric, and five repetition-grouped folds. The feature representation remained 200 ms windows: log RMS, MAV, waveform length; fusion adds accelerometer mean and standard deviation.

| Candidate | Event accuracy | Gesture accuracy | Tilt-only false commands | LEFT recall |
|---|---:|---:|---:|---:|
| Extra Trees + EMG/accel | 82.5% (99/120) | 78.75% (63/80) | 4/40 (10%) | 45% (9/20) |
| Random Forest + EMG/accel | 80.0% (96/120) | 75.0% (60/80) | 4/40 (10%) | 50% (10/20) |
| RBF SVM + EMG/accel | 75.0% (90/120) | 77.5% (62/80) | 12/40 (30%) | 45% (9/20) |
| Extra Trees + EMG only | 78.3% (94/120) | 71.25% (57/80) | 3/40 (7.5%) | 40% (8/20) |
| MLP + EMG/accel | 75.8% (91/120) | 72.5% (58/80) | 7/40 (17.5%) | 45% (9/20) |

Tree ensembles improved this sitting's candidate-screen scores, but model choice used these same folds, so the maximum is optimistic. All folds share the same placement/session. LEFT remains poor and the 40 tilt-only controls do not measure ordinary-life false commands per hour. This screen does not establish deployment accuracy or re-wear generalization. Full results: `session1_richer_models.json`; rerun with `within_session_richer_models.py`. The Extra Trees implementation is the scikit-learn randomized tree ensemble ([documentation](https://scikit-learn.org/stable/modules/ensemble.html#forest)); session/domain shifts require separate evaluation, as emphasized by [EMGBench](https://arxiv.org/abs/2410.23625).

### Window-duration screen (2026-11-03; exploratory)

The original 200 ms window was a baseline starting point. A fixed-fold sweep compared 100, 150, 200, 250, 300, and 400 ms windows, with a constant 20 ms hop, unchanged time-domain features, RBF SVM / Extra Trees, and EMG-only / fused modalities. Each repetition stayed within one fold, and the same event-level folds were used for all 24 configurations.

| Extra Trees candidate | Event accuracy | Gesture accuracy | Tilt-only false commands | LEFT recall |
|---|---:|---:|---:|---:|
| EMG + accel, 200 ms | 81.7% (98/120) | 77.5% (62/80) | 4/40 (10%) | 50% (10/20) |
| EMG + accel, 300 ms | 84.2% (101/120) | 80.0% (64/80) | 3/40 (7.5%) | 50% (10/20) |
| EMG + accel, 400 ms | 88.3% (106/120) | 85.0% (68/80) | 2/40 (5%) | 60% (12/20) |
| EMG only, 400 ms | 82.5% (99/120) | 82.5% (66/80) | 7/40 (17.5%) | 65% (13/20) |

Within this one-session event-vote screen, longer windows performed better for Extra Trees, particularly with fusion. This is not yet a live-window recommendation: majority votes use predictions across the full cued interval and do not measure action onset delay. The 400 ms candidate still missed 8/20 RIGHT intervals and 8/20 LEFT intervals, and repeated model/window screening on the same folds makes the top score optimistic. Window duration is task-dependent; prior work evaluates different ranges and describes 100–300 ms as a useful range in some movement-classification settings ([evaluation paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6946760/)). Full results and plot: `session1_window_duration_sweep.json`, `session1_window_duration_sweep.png`; rerun with `window_duration_sweep.py` and `plot_window_duration_sweep.py`. The sweep's fixed event-level folds differ from the earlier window-level grouped split, so compare candidates inside the sweep rather than against the previous report.

## References

- Hudgins, Parker & Scott (1993), foundational time-domain myoelectric pattern recognition: [PubMed](https://pubmed.ncbi.nlm.nih.gov/8468080/).
- `GroupKFold` keeps a complete group out of the opposite split: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- Overlapping EMG windows can leak signal across random splits: [dynamic EMG splitting study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9797837/).
- Project-specific review and collection guidance: [`semg_literature_review.md`](../../docs/semg_literature_review.md).
