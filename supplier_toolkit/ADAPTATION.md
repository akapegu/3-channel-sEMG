# Supplier toolkit adaptation

Upstream: https://github.com/upsidedownlabs/NPG-Lite-ArmBand-Software
Pinned revision: `1dc28c3ea755eaca5412f7255eb3010f9f6df142` (GPL-3.0).
Untouched source lives in `upstream/`; working changes in `adapted/`.
This is a separate environment from the tested raw recorder in `scripts/.venv`.

## Implemented

- Native 1000 SPS/channel acquisition, preserving the measured band stream.
- Shared causal filter implementation for recording/dashboard: Butterworth bandstop 48–52 Hz (order 2 prototype, resulting order 4), followed by order-2 70 Hz high-pass. This follows the supplier's intended notch/high-pass workflow; the source calls its high-pass an EMG bandpass. These settings are a supplier-compatible baseline, not proven optimal for this user's arm. No firmware change or downsampling.
- 500-sample windows / 250-sample stride retain the supplier's 0.5-second window / 50% overlap at the corrected rate.
- Complete original EMG and IMU notification bytes, characteristic IDs, monotonic and Unix arrival timestamps are streamed to `raw_packets/*.jsonl`, flushed on each notification. Every accelerometer frame is preserved there even though supplier model input still uses latest-value sample-and-hold. Processed training CSVs remain separate. No overwrite of existing CSV/model outputs.
- Processing-profile sidecars link processed recordings to raw logs. Trainer rejects mismatched profiles. Dashboard rejects old 500-SPS or synthetic-only models instead of silently using them.
- Mandatory explicit split manifest assigns whole acquisitions to train/validation/test before windows are created. A shared acquisition group cannot cross partitions. Training-only normalisation retained.
- Supplier CNN/LSTM/CNN-LSTM architectures and animated dashboard retained. Keyboard controller is disabled pending separate adaptation; no keyboard/HID output.
- Single-band scope; supplier's index-aligned multi-band merge is not treated as synchronised acquisition.

## Verified

`check_adaptation.py` checks per-sample filtering against SciPy SOS filtering, correct notch location, shared recording/live filters, native rate, byte-exact EMG/IMU preservation, metadata and rejection of cross-partition acquisition groups.

A synthetic 12-file, four-class dataset exercised the supplier CNN-LSTM trainer for one epoch with 28 windows in each partition. Model save/reload, reports and replay buffer worked. This is software verification only. Its model is marked `synthetic_only` and blocked from live dashboard inference. Synthetic scores are not wearable results.

The dashboard preview was visually checked and is explicitly labelled **LAYOUT PREVIEW — no model loaded; no sensor connection**. No personalised pinch/flexion/extension model has been trained. Adapted acquisition was subsequently verified live after recharge/re-wear: 5,140 EMG samples/channel and 560 accelerometer samples/axis, continuous received counters, and zero live-versus-raw-replay filter error. See `verification/livecheck03/results.json`. Personalised gesture acquisition/training and live inference remain pending.

## Commands

Run from the `supplier_toolkit/adapted` directory. Python executable is `../.venv/bin/python`.

```sh
../.venv/bin/python check_adaptation.py
../.venv/bin/python gesture_ui_server.py --preview --no_browser --port 8765
../.venv/bin/python record_gesture.py --max_record_seconds 30
```

Recording follows supplier prompts: subject, rest trials, gesture names (`pinch`, `flexion`, `extension`). Visual cues are now the default, per user preference: local timer with Start/Stop buttons, five-second preparation, named action and phase/trial countdowns. No sound is required. Optional `--cue_mode audio` retains the supplier beep path. Visual cue events/status are linked from processing metadata; aborted visual trials cannot silently enter training. Reviewed segment subsets can exclude incomplete holds while preserving original cue sidecars. Body position, cue compliance and actual contact conditions must be logged in the main experiment log. Battery-only while worn; browser visualizer disconnected. No new data capture is started by these instructions alone.

Collect separate acquisitions for each class in training, validation and test blocks; decide assignment before capture. Keep all files derived from one acquisition in the same group. Write a manifest relative to `training_data`, for example:

```json
{"files": [
  {"path":"subject/pinch/pinch_trial1_example.csv","partition":"train","acquisition_group":"block1"},
  {"path":"subject/pinch/pinch_trial2_example.csv","partition":"validation","acquisition_group":"block2"},
  {"path":"subject/pinch/pinch_trial3_example.csv","partition":"test","acquisition_group":"block3"}
]}
```

This is an illustrative fragment; an actual manifest must include every selected class/file, and every partition must contain every class. Prefer later-session test data; adjacent windows are correlated. Do not tune on test scores.

```sh
../.venv/bin/python train_gesture_model.py --data_dir training_data --split_manifest training_data/split_manifest.json --out_dir personalised_model_01 --arch cnn_lstm
../.venv/bin/python gesture_ui_server.py --model_dir personalised_model_01 --no_browser
```

Existing raw clench/tilt recordings are not silently relabelled as pinch/flexion/extension. Their nearest-centroid models remain separate historical experiments. No clinical calibration, gesture accuracy, robust latency or BLE HID capability is established by setup verification.

## Remaining limitations

Raw notification arrival times are not acquisition timestamps. Supplier processed CSV timestamps are inferred by counting backward from packet arrival; they are not precise synchronisation. Actual sensor rate tolerance, motion sample-hold timing, cue reaction delay and startup filter transients remain relevant. Recording and training use the new processing profile; bundled upstream model metadata is retained for reference only, without presenting its reported accuracy as independent evidence.
