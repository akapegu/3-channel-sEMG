"""Loopback-only live prediction display for the locked EXP-015 swipe model.

The page shows per-window and short majority-vote predictions while a
continuous BLE take is preserved. It sends no keyboard, HID, or device
control actions other than starting/stopping the band data stream.
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
from bleak import BleakScanner

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP_DIR = ROOT / "analysis" / "exp015_swipe4"
OUT_DIR = ROOT / "analysis" / "exp016_swipe_shadow"
MODEL_PATH = EXP_DIR / "model_v1" / "locked_model.joblib"
MODEL_META_PATH = EXP_DIR / "model_v1" / "locked_model.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(HERE)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP_DIR))

import record_gesture as recorder  # noqa: E402
from train_swipe4_baseline import WINDOW, HOP, window_features  # noqa: E402

recorder.SAMPLING_RATE = 1000
MODEL_META = json.loads(MODEL_META_PATH.read_text())
if MODEL_META.get("selected_candidate") != "LDA_FUSION":
    raise RuntimeError("This live bridge is written for the locked LDA_FUSION candidate")
MODEL = joblib.load(MODEL_PATH)
CLASSES = [str(c) for c in MODEL.classes_]
if set(CLASSES) != {"rest", "down", "left", "right", "top"}:
    raise RuntimeError(f"Unexpected model classes: {CLASSES}")

LOCK = threading.Lock()
STATE = {
    "phase": "READY", "message": "Ready. Settle your arm, then press Start live shadow mode.",
    "connected": False, "device": None, "samples": 0, "elapsed_s": 0.0,
    "window_count": 0, "window_ms": WINDOW, "update_ms": HOP,
    "current_class": "warming up", "stable_class": "warming up",
    "stable_rule": "majority of the latest 5 window votes; at least 3 needed",
    "score": None, "scores": {}, "channel_rms": [], "transitions": 0,
    "raw_packet_path": None, "signal_csv": None, "prediction_csv": None,
    "summary_path": None, "error": None,
}
LOOP = None
START_REQUEST = None
STOP_REQUEST = None
RECORDER = None


def update(**fields):
    with LOCK:
        STATE.update(fields)


def snapshot():
    with LOCK:
        return json.loads(json.dumps(STATE))


PAGE = r'''<!doctype html><meta charset="utf-8"><title>Live swipe classifier</title>
<style>
body{font:20px system-ui;background:#111827;color:#f3f4f6;text-align:center;padding:20px;margin:0}
h1{font-size:40px;margin:12px}.card{background:#1f2937;border-radius:14px;padding:18px;margin:14px auto;max-width:920px}
#stable{font-size:54px;font-weight:750;min-height:68px}.raw{font-size:22px;color:#cbd5e1}
button{font:inherit;padding:13px 26px;margin:6px;border:0;border-radius:8px;cursor:pointer}
#start{background:#22c55e;color:#082f18}#stop{background:#f87171;color:#450a0a}
.barrow{display:grid;grid-template-columns:90px 1fr 60px;gap:10px;align-items:center;margin:10px auto;max-width:720px;text-align:left}
.track{height:20px;background:#374151;border-radius:10px;overflow:hidden}.bar{height:100%;width:0;background:#38bdf8;transition:width .08s linear}
.small{font-size:15px;color:#cbd5e1;max-width:850px;margin:12px auto}.status{font-size:16px;color:#93c5fd}
.metrics{display:flex;gap:26px;justify-content:center;flex-wrap:wrap;font-size:17px;color:#d1d5db}
</style>
<h1>Live four-swipe model</h1>
<div class="card"><div id="phase">READY</div><div id="stable">—</div>
<div class="raw">Current 200 ms window: <b id="raw">—</b> · model score <b id="score">—</b></div>
<div class="metrics"><span>BLE: <b id="ble">not connected</b></span><span>Samples: <b id="samples">0</b></span>
<span>Elapsed: <b id="elapsed">0.0 s</b></span><span>Window votes: <b id="windows">0</b></span>
<span>Rest-to-swipe transitions: <b id="transitions">0</b></span></div>
<p id="message" class="status">Ready. No recording is active.</p>
<button id="start">Start live shadow mode</button><button id="stop" hidden>Stop and save</button>
</div>
<div class="card"><h2>Current model outputs</h2><div id="bars"></div>
<p class="small">Displayed values are LDA model probabilities, not yet calibrated as reliable confidence. The large label is a 3-of-5 vote over the latest 200 ms-window predictions; the raw label above updates every 20 ms of new samples.</p>
</div>
<div class="card"><p><b>Try it:</b> relax for a few seconds, then do one familiar swipe at a time with a relaxed pause between them. Watch whether the class changes to the movement you just made and returns to REST.</p>
<p class="small">This is shadow mode: it records the band and shows predictions. It does not press keys, control BLE HID, or trigger other devices. Press Stop to preserve this take and its prediction timeline.</p></div>
<script>
const names=['rest','down','left','right','top'];
const bars=document.getElementById('bars');
for(const n of names){const row=document.createElement('div');row.className='barrow';row.innerHTML=`<span>${n.toUpperCase()}</span><div class="track"><div class="bar" id="bar-${n}"></div></div><span id="pct-${n}">—</span>`;bars.appendChild(row);}
const start=document.getElementById('start'),stop=document.getElementById('stop');
start.onclick=async()=>{start.disabled=true;try{const r=await fetch('/api/start',{method:'POST'});if(!r.ok)throw new Error(await r.text());}catch(e){start.disabled=false;document.getElementById('message').textContent='Start failed: '+e;}};
stop.onclick=async()=>{stop.disabled=true;await fetch('/api/stop',{method:'POST'});};
async function tick(){try{const r=await fetch('/api/state',{cache:'no-store'}),s=await r.json();
document.getElementById('phase').textContent=s.phase;document.getElementById('stable').textContent=s.stable_class.toUpperCase();
document.getElementById('raw').textContent=s.current_class;document.getElementById('score').textContent=s.score==null?'—':(100*s.score).toFixed(1)+'%';
document.getElementById('ble').textContent=s.connected?'connected · '+s.device:'not connected';
document.getElementById('samples').textContent=Number(s.samples).toLocaleString();document.getElementById('elapsed').textContent=Number(s.elapsed_s).toFixed(1)+' s';
document.getElementById('windows').textContent=Number(s.window_count).toLocaleString();document.getElementById('transitions').textContent=s.transitions;
document.getElementById('message').textContent=s.message;
for(const n of names){const v=(s.scores&&s.scores[n])||0;document.getElementById('bar-'+n).style.width=(100*v)+'%';document.getElementById('pct-'+n).textContent=(100*v).toFixed(1)+'%';}
start.hidden=!['READY','DONE','ERROR'].includes(s.phase);start.disabled=false;stop.hidden=!['CONNECTING','STREAMING','STOPPING'].includes(s.phase);
}catch(e){document.getElementById('message').textContent='Local live service unavailable: '+e;}
setTimeout(tick,75);}tick();
</script>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            return self.send_json(snapshot())
        if path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        host = self.headers.get("Host", "")
        if (host != f"127.0.0.1:{SERVER.server_port}" or
                self.headers.get("Origin") != f"http://{host}"):
            return self.send_json({"error": "Local-origin request required"}, 403)
        if path == "/api/start":
            if snapshot()["phase"] not in ("READY", "DONE", "ERROR"):
                return self.send_json({"error": "A live session is already active"}, 409)
            STOP_REQUEST.clear()
            LOOP.call_soon_threadsafe(START_REQUEST.set)
            return self.send_json({"ok": True})
        if path == "/api/stop":
            STOP_REQUEST.set()
            return self.send_json({"ok": True})
        return self.send_json({"error": "Not found"}, 404)


async def live_session():
    global RECORDER
    update(phase="SCANNING", message="Scanning for the NPG-Lite band…", connected=False,
           samples=0, elapsed_s=0.0, window_count=0, current_class="warming up",
           stable_class="warming up", score=None, scores={}, channel_rms=[], transitions=0,
           raw_packet_path=None, signal_csv=None, prediction_csv=None, summary_path=None, error=None)
    rec = None
    prediction_path = None
    summary_path = None
    source_path = None
    prediction_count = 0
    status = "stopped"
    failure = None
    started_unix_ns = time.time_ns()
    try:
        found = await BleakScanner.discover(timeout=8, return_adv=True)
        matches = []
        for device, adv in found.values():
            name = (adv.local_name or device.name or "").lower()
            address = (device.address or "").lower()
            if recorder.device_name_matches(name) and ("27:bc" in name or "27:bc" in address):
                matches.append(device)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one band ending 27:BC; scan found {len(matches)}. Ensure it is powered on and not connected elsewhere.")
        device = matches[0]
        rec = recorder.DeviceRecorder(device, out_path=None, channels=3, role="swipe")
        RECORDER = rec
        update(phase="CONNECTING", message=f"Connecting to {rec.name}…", device=rec.name,
               raw_packet_path=rec.raw_path)
        await rec.connect()
        await rec.start_streaming()

        prediction_dir = OUT_DIR / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns()}"
        prediction_path = prediction_dir / f"swipe_shadow_predictions_{stamp}.csv"
        columns = ["host_unix_ns", "elapsed_s", "sample_end_exclusive", "raw_class",
                   "stable_class", "top_score", *[f"score_{c}" for c in CLASSES]]
        prediction_file = prediction_path.open("x", newline="")
        writer = csv.writer(prediction_file)
        writer.writerow(columns)
        prediction_file.flush()

        origin = time.perf_counter()
        started_unix_ns = time.time_ns()
        seen_rows = 0
        buffer = deque(maxlen=WINDOW)
        since_prediction = 0
        recent_votes = deque(maxlen=5)
        previous_stable = "rest"
        transition_count = 0
        last_data_at = origin
        last_flush = origin
        update(phase="STREAMING", connected=True,
               message="Connected and recording. Relax briefly, then make one familiar swipe at a time.")

        try:
            while True:
                await asyncio.sleep(.02)
                if not rec.client or not rec.client.is_connected:
                    raise ConnectionError("BLE connection dropped")
                new_rows = len(rec.rows)
                if new_rows > seen_rows:
                    last_data_at = time.perf_counter()
                    for row_index in range(seen_rows, new_rows):
                        row = rec.rows[row_index]
                        sample = np.asarray(row[2:8], dtype=np.float64)
                        buffer.append(sample)
                        seen_rows = row_index + 1
                        since_prediction += 1
                        if len(buffer) < WINDOW or since_prediction < HOP:
                            continue
                        since_prediction = 0
                        window = np.asarray(buffer, dtype=np.float64)
                        feature = window_features(window, use_accel=True)
                        feature_row = feature.reshape(1, -1)
                        scores_arr = MODEL.predict_proba(feature_row)[0]
                        class_index = int(np.argmax(scores_arr))
                        raw_class = CLASSES[class_index]
                        top_score = float(scores_arr[class_index])
                        recent_votes.append(raw_class)
                        counts = Counter(recent_votes)
                        winner, votes = counts.most_common(1)[0]
                        stable_class = winner if len(recent_votes) == 5 and votes >= 3 else "mixed"
                        if stable_class not in ("mixed", "rest") and previous_stable == "rest":
                            transition_count += 1
                        if stable_class != "mixed":
                            previous_stable = stable_class
                        rms = np.sqrt(np.mean((window[:, :3] - np.mean(window[:, :3], axis=0)) ** 2, axis=0))
                        elapsed = time.perf_counter() - origin
                        unix_ns = time.time_ns()
                        score_map = {c: float(scores_arr[i]) for i, c in enumerate(CLASSES)}
                        writer.writerow([unix_ns, round(elapsed, 4), seen_rows, raw_class,
                                         stable_class, top_score, *[score_map[c] for c in CLASSES]])
                        prediction_count += 1
                        if prediction_count % 50 == 0:
                            prediction_file.flush()
                        update(samples=seen_rows, elapsed_s=round(elapsed, 2),
                               window_count=prediction_count, current_class=raw_class,
                               stable_class=stable_class, score=top_score, scores=score_map,
                               channel_rms=[float(v) for v in rms], transitions=transition_count,
                               prediction_csv=str(prediction_path), connected=True)
                elapsed = time.perf_counter() - origin
                update(samples=len(rec.rows), elapsed_s=round(elapsed, 2))
                if time.perf_counter() - last_data_at > 3.0:
                    raise TimeoutError("No EMG samples received for 3 seconds")
                if STOP_REQUEST.is_set():
                    status = "user_stopped"
                    break
                if time.perf_counter() - last_flush >= 1.0:
                    prediction_file.flush()
                    last_flush = time.perf_counter()
        finally:
            prediction_file.flush()
            prediction_file.close()

    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        status = "error"
        update(phase="ERROR", error=failure, message=f"Live session stopped: {failure}")
    finally:
        if rec is not None:
            rec.visual_cue_status = status
            try:
                await rec.stop_and_disconnect()
            except Exception as exc:
                if failure is None:
                    failure = f"cleanup: {type(exc).__name__}: {exc}"
                    status = "error"
            if rec.rows:
                source_dir = OUT_DIR / "signals"
                source_dir.mkdir(parents=True, exist_ok=True)
                trial = len(list(source_dir.glob("swipe_shadow_trial*_merged_*.csv"))) + 1
                saved = recorder.merge_and_save(str(source_dir), "swipe_shadow", trial,
                                                [rec], reps=[], mode="reps")
                source_path = saved["csv_path"] if saved else None
            if not rec.raw_file.closed:
                rec.raw_file.close()
        summary_path = OUT_DIR / f"swipe_shadow_session_{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns()}.json"
        summary = {
            "status": status, "error": failure, "device": rec.name if rec else None,
            "started_host_unix_ns": started_unix_ns,
            "sampling_rate_configured_hz": 1000,
            "preprocessing_profile": MODEL_META["feature_config"].get("preprocessing_profile", "npg-native1000-bs48_52-hp70-v1"),
            "model_path": str(MODEL_PATH), "model_candidate": MODEL_META["selected_candidate"],
            "class_order": CLASSES, "window_samples": WINDOW, "hop_samples": HOP,
            "processed_samples": len(rec.rows) if rec else 0,
            "prediction_windows": prediction_count,
            "stable_rule": "majority among last five predictions, at least three votes; otherwise mixed",
            "raw_packet_file": rec.raw_path if rec else None,
            "filtered_signal_csv": source_path,
            "prediction_csv": str(prediction_path) if prediction_path else None,
            "note": "Shadow-mode predictions only. No keyboard/HID actions. Per-class LDA scores are not calibration-validated. No ground-truth movement markers were recorded in this continuous exploratory take.",
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        update(phase="DONE" if status == "user_stopped" else "ERROR",
               connected=False, samples=len(rec.rows) if rec else 0,
               current_class="—", stable_class="—",
               message=(f"Stopped and saved the take. {prediction_count:,} live windows classified."
                        if status == "user_stopped" else f"Session ended: {failure or status}"),
               raw_packet_path=rec.raw_path if rec else None,
               signal_csv=source_path, prediction_csv=str(prediction_path) if prediction_path else None,
               summary_path=str(summary_path), error=failure)
        RECORDER = None


async def main():
    global LOOP, START_REQUEST, STOP_REQUEST, SERVER
    LOOP = asyncio.get_running_loop()
    START_REQUEST = asyncio.Event()
    STOP_REQUEST = threading.Event()
    SERVER = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=SERVER.serve_forever, daemon=True).start()
    print(f"LIVE SWIPE SHADOW PAGE: http://127.0.0.1:{SERVER.server_port}/", flush=True)
    print(f"Locked model: {MODEL_PATH}", flush=True)
    try:
        while True:
            await START_REQUEST.wait()
            START_REQUEST.clear()
            await live_session()
    finally:
        SERVER.shutdown()
        SERVER.server_close()


if __name__ == "__main__":
    asyncio.run(main())
