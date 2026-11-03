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

### Session 1 → Session 2 re-wear evaluation (2026-11-03; development)

Session 2 was recorded after removing and re-wearing the band, approximately 39 minutes after Session 1 on the same day. QC found 512,580 EMG samples, continuous counters, finite processed values, and all 120 action cues in bounds: 20 per gesture plus 10 relaxed-finger tilt controls in each direction. The participant confirmed following the complete sequence. Source CSV, exact BLE packet bytes, cue log, and capture metadata remain local and ignored by Git.

Candidates were trained only on Session 1 and scored on Session 2; no Session 2 rows were used to fit these models. Extra Trees with EMG+accelerometer features at 400 ms reached **107/120 event intervals (89.2%)**, **68/80 gestures (85.0%)**, and **1/40 tilt-only false commands (2.5%)**. Per-class recall was REST 39/40, DOWN 18/20, LEFT 19/20, RIGHT 14/20, TOP 17/20. The 300 ms fused Extra Trees model reached 98/120 events, 60/80 gestures, and 2/40 tilt false commands. EMG-only Extra Trees at 400 ms had zero tilt false commands but only 46/80 gestures correct (57.5%), especially weak RIGHT (6/20). The RBF SVM with fusion collapsed mostly to LEFT and falsely called 35/40 tilt controls commands, showing that the result is model-family specific.

This is encouraging cross-placement evidence, but it is still a development comparison: the band was re-worn on the same day, there is one training wearing, the candidates were screened on Session 2, and majority votes cover the full prompted interval rather than measuring onset delay. Keep Session 3 for further development/confirmation and Session 4 untouched for the final test. Full metrics: `session2_cross_session_models.json`; rerun with `evaluate_cross_session_window_candidates.py` by supplying local training CSV/cue and test CSV/cue paths. Cross-session evaluation is a central concern in sEMG benchmarks such as [EMGBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59fe60482e2e5faf557c37d121994663-Abstract-Datasets_and_Benchmarks_Track.html).

### Session 1+2 → Session 3 fixed-candidate confirmation (2026-11-04)

Session 3 is a complete re-wear capture with 512,580 samples and 120 in-bounds action cues (20 per gesture plus 40 relaxed-finger tilt controls). EMG and accelerometer counters are continuous; no malformed/empty packets were found; processed values are finite. Host-arrival estimates were 1000.058 EMG frames/s and 103.833 accelerometer frames/s. Raw CSV, packet bytes, and cue metadata remain local and ignored by Git.

The candidate selected in the Session 2 screen was fixed: 400 ms window, 20 ms hop, the same time-domain EMG features plus accelerometer summaries, and Extra Trees. It was trained on Sessions 1+2 and scored on Session 3 without tuning there. It classified 102/120 intervals (85.0%), including 62/80 gestures (77.5%), and made 0/40 false commands on the tilt-only controls. Recall: DOWN 16/20 (80%), LEFT 16/20 (80%), RIGHT 17/20 (85%), TOP 13/20 (65%), REST 40/40. All seven missed TOP cues were predicted REST. This does not meet the project's 90% event-accuracy target; zero false tilt commands in 40 controls does not establish a low real-world false-command rate.

This confirms some cross-wear transfer, but Session 3 is still development data. It is the third wearing for one user on nearby dates; cue-level majority votes do not measure onset latency, and the target error is TOP→REST. Before Session 4, run session-held-out model/window validation across Sessions 1–3, inspect TOP trials, choose/freeze one pipeline, then collect Session 4 as the untouched final evaluation. Full metrics and reproducible entry point: `session3_confirmation.json` and `evaluate_session3_confirmation.py`.

### Session-held-out validation across Sessions 1–3 (2026-11-04)

Ran three folds: in each fold, fit on two complete sessions and score on the third. This uses all sessions across folds but no single validation model sees all three. The five-candidate screen favored Extra Trees with EMG+accelerometer, 400 ms windows, and a 20 ms hop.

| Held-out session | Event accuracy | Gesture accuracy | Tilt-only false commands | TOP recall |
|---|---:|---:|---:|---:|
| Session 1 | 102/120 (85.0%) | 66/80 (82.5%) | 4/40 (10%) | 20/20 (100%) |
| Session 2 | 101/120 (84.2%) | 61/80 (76.25%) | 0/40 | 16/20 (80%) |
| Session 3 | 102/120 (85.0%) | 62/80 (77.5%) | 0/40 | 13/20 (65%) |
| Pooled out-of-session | 305/360 (84.7%) | 189/240 (78.75%) | 4/120 (3.3%) | 49/60 (81.7%) |

All 11 missed TOP events in Sessions 2–3 were predicted REST. Their median filtered EMG RMS was lower than correctly recognized TOP trials, especially on channels 1 and 2, and approached tilt-control values on those channels. This is descriptive evidence that the weak TOP attempts may be less muscle-active or differ after re-wearing; it does not prove why. Candidate comparisons: ET fusion 300 ms, 79.7% event / 71.25% gesture; ET fusion 200 ms, 71.1% / 57.1%; ET EMG-only 400 ms, 76.7% / 68.75%; RBF SVM fusion 400 ms, 62.5% / 60.8% with 41/120 false tilt commands.

This screen is below the project's 90% event-accuracy and zero-tilt-false-command targets. Candidate selection used these same three folds, so pooled results are development estimates and may be optimistic. Leave-one-session-out results and error detail: `session1_3_leave_one_session_out.json`; rerun with `evaluate_leave_one_session_out.py` using explicit local paths.

After the comparison, the provisional Extra Trees fusion 400 ms candidate was fit on all Sessions 1–3 (20,145 training windows) for Session 4 evaluation. Its training fit was not a validation score. Local artifacts are in `model_session123/`; they are excluded from Git. The reusable fit command is `train_session123_candidate.py`. Session 4 has since been evaluated once as the locked final test; results are below. Cross-session robustness motivated holding out complete sessions, as in [EMGBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59fe60482e2e5faf557c37d121994663-Abstract-Datasets_and_Benchmarks_Track.html).

### Locked Session 4 final evaluation (2026-11-04)

The frozen Extra Trees fusion candidate (trained on Sessions 1–3; 400 ms features, 20 ms hop) was scored once on Session 4. Session 4 was not used for fitting or tuning. The complete re-wear capture contains 512,640 samples and all 120 in-bounds cues. Packet length, sample counters, finite processed values, and both sensor streams passed capture QC; host-arrival estimates were 1000.052 EMG frames/s and 103.832 accelerometer frames/s.

| Session 4 measure | Result |
|---|---:|
| Action interval accuracy | 68/120 (56.7%) |
| Intended gesture accuracy | 46/80 (57.5%) |
| Tilt-only controls correctly rejected | 22/40 (55.0%) |
| False commands on tilt-only controls | 18/40 (45.0%) |
| DOWN / LEFT / RIGHT / TOP recall | 50% / 50% / 100% / 30% |

The 18 false commands were six each during LEFT-, RIGHT-, and TOP-directed tilts; none occurred during DOWN-directed tilts. Fifteen of 40 REST controls were predicted RIGHT. This is a failed robustness result relative to both the project target and the 84.7% event accuracy / 3.3% false tilt rate from the development leave-one-session-out screen. The model's high RIGHT recall is not enough to offset its false activations.

Raw-packet inspection found exact-zero ADC values on CH2 in 616/512,640 samples (0.120%) and CH3 in 300/512,640 (0.059%); the longest runs were 19 ms and 15 ms. CH1 had none. Sessions 1–3 had no exact-zero EMG samples. Session 4 channel medians were [1266, 1278, 1283] counts, near the earlier sessions' approximate 1280-count medians. These sparse zeros are an anomaly of unknown cause, but do not show a sustained channel outage. They do not by themselves explain or invalidate the low score.

Interpretation is limited: the model did not generalize well to this re-wear, while this one capture cannot identify whether placement, execution variability, distribution shift, or model design caused the errors. The Session 4 test has now been consumed. Do not tune against it and reuse it as a final test. Add new sessions to development, revise and evaluate candidates with complete sessions held out, then collect a new untouched final session. This emphasis on session-wise evaluation reflects known sEMG distribution/repeatability concerns ([EMGBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59fe60482e2e5faf557c37d121994663-Abstract-Datasets_and_Benchmarks_Track.html); [NinaPro DB6 repeatability protocol](https://ninapro.hevs.ch/instructions/DB6.html)).

Detailed event and window metrics: `session4_locked_test.json`. Reproduce the one-time frozen-model scoring with `evaluate_session4_locked.py`; raw data, cue metadata, and model weights remain local and ignored by Git.

### Supplier-style CNN-LSTM screen across Sessions 1–4 (2026-11-04)

Trained the supplier repository's CNN→LSTM architecture on the four personalized re-wear captures. The architecture uses two Conv1D blocks followed by 64- and 32-unit LSTM layers ([supplier model source](https://github.com/upsidedownlabs/NPG-Lite-ArmBand-Software/blob/main/model_architectures.py#L66-L89)). Input windows were 500 ms with 250 ms hop, matching the supplier's nominal 500 ms window and 50% overlap at this project's 1,000 Hz stream rate. Compared three EMG channels alone against EMG plus the LIS3DH axes.

Validation held out one complete wearing per fold. Within the three training wearings, 20% of complete cued repetitions per class were held out for early stopping; no overlapping windows from a cue were split between fit and validation. Window normalization was fit on the fold's training cues only. The model and protocol therefore differ from the supplier training script's default random split of already-overlapping windows ([windowing code](https://github.com/upsidedownlabs/NPG-Lite-ArmBand-Software/blob/main/train_gesture_model.py#L232-L252), [random train/validation/test split](https://github.com/upsidedownlabs/NPG-Lite-ArmBand-Software/blob/main/train_gesture_model.py#L376-L392)); that script does not hold out complete wearings in its default path.

| Input | Held-session event accuracy | Intended gesture accuracy | Tilt-only false commands |
|---|---:|---:|---:|
| EMG only | 361/480 (75.2%) | 217/320 (67.8%) | 16/160 (10.0%) |
| EMG + accelerometer | 313/480 (65.2%) | 155/320 (48.4%) | 2/160 (1.25%) |

| Held-out wearing | EMG-only gesture accuracy / false tilt commands | Fused gesture accuracy / false tilt commands |
|---|---:|---:|
| Session 1 | 49/80 (61.3%) / 0/40 | 40/80 (50.0%) / 0/40 |
| Session 2 | 58/80 (72.5%) / 0/40 | 32/80 (40.0%) / 0/40 |
| Session 3 | 60/80 (75.0%) / 1/40 | 55/80 (68.8%) / 0/40 |
| Session 4 | 50/80 (62.5%) / 15/40 | 28/80 (35.0%) / 2/40 |

EMG-only had the higher gesture recall, but its Session 4 errors are not acceptable: it recognized only 1/20 TOP repetitions and generated 15 false commands in 40 tilt controls. Fusion suppressed most tilt controls but recognized fewer than half the gesture repetitions overall. Neither candidate meets the intended-control requirement. The modality choice was made using these same four folds, so pooled scores are development estimates, not independent final accuracy. Session 4 has already been examined and is no longer a clean final test.

The all-session EMG-only development artifact is local/ignored at `cnn_lstm_sessions/cnn_lstm_all_sessions.keras`; its exact scaling values and fold predictions are in the ignored `cnn_lstm_sessions/cnn_lstm_session_loso.json`. The artifact was trained for 29 epochs using 1,847 windows, with 467 complete-cue validation windows used for early stopping. It is for offline/shadow development only. Reproduce the comparison with `train_cnn_lstm_sessions.py` and the installed `supplier_toolkit/upstream/requirements.txt` dependencies.

### Planned next steps

1. Audit the CNN-LSTM's Session 4 errors descriptively, starting with TOP→LEFT (18/20) and tilt-control→LEFT/DOWN (15/40 total). Compare the labeled cue traces and per-channel EMG features across all four wearings. This analysis may explain what to investigate, but Session 4 is already development data and cannot provide another independent score.
2. Keep the four movement labels unchanged during that audit. If the labeled actions are not physically repeatable/distinct, revise the gesture definitions before collecting more labeled trials. If they are repeatable, record Sessions 5 and 6 as development re-wearings with the same 20 repetitions per gesture and 10 relaxed-finger tilt controls per direction; note placement/orientation changes. Keep EMG-only and fusion candidates separate and score by whole-session holdout.
3. Freeze one pipeline using Sessions 1–6, then record Session 7 as a new locked evaluation. Include an ordinary no-command activity block in addition to tilt-only controls, because the current 40 REST controls per wearing measure prompted tilts rather than daily false-command rate. This session-wise plan reflects the distribution-shift evaluation emphasized in [EMGBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/59fe60482e2e5faf557c37d121994663-Abstract-Datasets_and_Benchmarks_Track.html); NinaPro DB6 studies repeatability with repeated acquisitions over days ([protocol](https://ninapro.hevs.ch/instructions/DB6.html)).

## References

- Hudgins, Parker & Scott (1993), foundational time-domain myoelectric pattern recognition: [PubMed](https://pubmed.ncbi.nlm.nih.gov/8468080/).
- `GroupKFold` keeps a complete group out of the opposite split: [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).
- Overlapping EMG windows can leak signal across random splits: [dynamic EMG splitting study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9797837/).
- Project-specific review and collection guidance: [`semg_literature_review.md`](../../docs/semg_literature_review.md).
