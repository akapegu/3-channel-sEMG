# Single-fist EMG flap prototype

This page supports both offline replay and a **live wearable prototype**. It uses the newer ChordsWeb recording for replay, or filtered Channel 1 from the band for live clench events.

## Input and detector

1. The source CSV is `recordings/NPG-Lite-Band-3CH-27-BC-D1-ChordsWeb-20261101-044735.csv`. It contains processed floating-point browser values and an 8-bit sample counter, not untouched ADC counts. The original file is read-only.
2. `analyse_repeated_fist.py` reads Channel 1 and computes mean-centered RMS over 100 samples (about 100 ms) every 50 samples. This reduces the fast EMG waveform to a slower activation envelope.
3. The first two seconds are assumed relaxed. The high threshold is `max(0.05, baseline median + 8 × scaled MAD)`; the release threshold is 55% of the high threshold. For this CSV those are 0.050 and 0.0275 processed units.
4. The detector emits one event after two consecutive envelope windows above the high threshold. It re-arms after three consecutive windows below the release threshold and enforces a 400 ms refractory interval. These rules suppress short spikes and duplicate triggers during one clench.
5. On this take the detector emits 19 **candidate** events. They have not been checked against an externally marked attempt list, so this is not yet a measured hit or false-trigger rate.

## Game

The offline replay server is `http://127.0.0.1:8766/flap_replay.html`; it loads `channel1_envelope.csv` and `summary.json` locally. It re-runs the high/low/re-arm logic in JavaScript and sends one canvas-game flap for each event.

For live mode, start `supplier_toolkit/adapted/live_flap_server.py` from this project using its virtual environment, then open `http://127.0.0.1:8767/flap_replay.html`. The server binds to loopback only. BLE scanning and connection start only after pressing **Connect & play live**. Keep relaxed for the first two seconds of calibration, then make a fist from relaxed and release between attempts. Press **Stop live recording** to stop and save. Port 8766 remains offline-only.

The bridge preserves every BLE notification byte-for-byte in `supplier_toolkit/adapted/raw_packets/*.jsonl` and writes a raw ADC CSV with all three untouched channels and counters. It also saves the recorder-filtered three EMG channels and sample-held accelerometer values under `analysis/exp012_live_flap/`. Live filtering uses the existing Python 50 Hz notch and EMG bandpass configuration. Channel 1's mean-centered RMS uses 100 samples per window and a 50-sample hop. Thresholds are estimated from the first two seconds of relaxed live baseline (median plus eight scaled MADs, with a numeric floor); two consecutive high windows trigger, three low windows re-arm, and a 400 ms refractory interval suppresses repeats. These are provisional assumptions pending labeled event comparison.

`Manual play` is a keyboard/click fallback for checking the game itself. It does not exercise the EMG detector. The game has no external assets, APIs, or network calls.

## Interpretation and next step

Replay demonstrates the software path `processed browser EMG → envelope → threshold event → flap`. Live mode adds `BLE raw notifications → preserved ADC counts → Python notch/bandpass → live envelope → calibrated threshold → flap`. This first live attempt is for checking connection, rate/counter continuity, and whether the fist can drive play; it does not establish robust gesture detection or safe device control. Record intended clench times so extra and missed triggers can be assessed.
