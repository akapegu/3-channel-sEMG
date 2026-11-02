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

## References

- Hudgins, Parker & Scott (1993), foundational time-domain myoelectric pattern recognition: [PubMed](https://pubmed.ncbi.nlm.nih.gov/8468080/).
- `GroupKFold` keeps a complete group out of the opposite split: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- Overlapping EMG windows can leak signal across random splits: [dynamic EMG splitting study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9797837/).
- Project-specific review and collection guidance: [`semg_literature_review.md`](../../docs/semg_literature_review.md).
