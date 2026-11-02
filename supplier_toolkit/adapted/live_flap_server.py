"""Loopback-only BLE bridge for the fist-clench flap game.

The browser asks this process to start; BLE discovery/connection happens only
after that button press. DeviceRecorder preserves every notification as raw
JSONL and runs the repository's established notch + EMG filters.
"""
from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import statistics
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GAME_DIR = ROOT / "analysis" / "exp011_single_fist_flap"
OUT_DIR = ROOT / "analysis" / "exp012_live_flap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(HERE)
sys.path.insert(0, str(HERE))
import record_gesture as recorder_module  # noqa: E402

STATE_LOCK = threading.Lock()
STATE = {
    "phase": "READY", "message": "Ready. Settle your arm, then press Connect & play live.",
    "samples": 0, "elapsed_s": 0.0, "high": None, "low": None,
    "envelope": [], "events": [], "raw_path": None, "csv_path": None,
    "error": None, "device": None,
}
START_REQUEST = None
STOP_REQUEST = None
LOOP = None
RECORDER = None
ADC_ROWS = []

# Live response settings at the configured nominal 1000 samples/second.
# A 20-sample window means 20 ms of EMG; evaluate every 10 new samples.
DETECT_WINDOW_SAMPLES = 20
DETECT_HOP_SAMPLES = 10


def update(**fields):
    with STATE_LOCK:
        STATE.update(fields)


def snapshot():
    with STATE_LOCK:
        return dict(STATE)


class RawRecorder(recorder_module.DeviceRecorder):
    """Keep a convenient raw ADC CSV alongside the byte-exact JSONL packets."""
    def __init__(self, *args, **kwargs):
        self.raw_adc_rows = []
        self.capture_start = time.perf_counter()
        super().__init__(*args, **kwargs)

    def handle_notify(self, sender, data):
        if len(data) == recorder_module.SAMPLES_PER_PACKET * 7:
            offset = 0
            packet_end = time.perf_counter()
            for i in range(recorder_module.SAMPLES_PER_PACKET):
                counter = data[offset]
                vals = [((data[offset + 1 + c * 2] << 8) | data[offset + 2 + c * 2])
                        for c in range(3)]
                offset += 7
                self.raw_adc_rows.append([packet_end - self.capture_start -
                    (recorder_module.SAMPLES_PER_PACKET - 1 - i) / recorder_module.SAMPLING_RATE,
                    counter, *vals])
        super().handle_notify(sender, data)

    def save_raw_adc(self):
        path = OUT_DIR / f"live_raw_adc_{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns()}.csv"
        with path.open("x", newline="") as f:
            w = csv.writer(f)
            w.writerow(["elapsed_s", "counter", "adc1", "adc2", "adc3"])
            w.writerows(self.raw_adc_rows)
        return str(path)

    def save_filtered(self):
        path = OUT_DIR / f"live_filtered_{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns()}.csv"
        with path.open("x", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "counter", "ch1", "ch2", "ch3",
                        "accel_x", "accel_y", "accel_z"])
            w.writerows(self.rows)
        return str(path)


def robust_threshold(values):
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values)
    sigma = 1.4826 * mad
    # A small absolute floor prevents numerical-zero noise from causing
    # every ordinary fluctuation to count as a gesture.
    high = max(med + 4.0 * sigma, med * 1.8, 1e-6)
    return high, 0.55 * high


async def live_run():
    global RECORDER
    update(phase="SCANNING", message="Scanning for the NPG-Lite band…", error=None,
           samples=0, elapsed_s=0.0, high=None, low=None, envelope=[], events=[],
           raw_path=None, csv_path=None, device=None)
    try:
        devices = await recorder_module.BleakScanner.discover(timeout=8)
        found = [d for d in devices if recorder_module.device_name_matches(d.name)
                 and ("27:bc" in d.address.lower() or "27:bc" in (d.name or "").lower())]
        if len(found) != 1:
            raise RuntimeError(f"Expected one band ending 27:BC; scan found {len(found)} matching devices. Ensure it is powered on and not connected elsewhere.")
        dev = found[0]
        RECORDER = RawRecorder(dev, "unused.csv", channels=3, role="npg_lite")
        update(phase="CONNECTING", message=f"Connecting to {dev.name or dev.address}…",
               raw_path=str(RECORDER.raw_path), device=dev.name or dev.address)
        await RECORDER.connect()
        await RECORDER.start_streaming()
        origin = time.perf_counter()
        update(phase="CALIBRATING", message="Connected. Keep your hand relaxed for the 2-second baseline…")

        seen_rows = 0
        last_packet_at = origin
        processed = []
        points = []
        event_no = 0
        armed = True
        above = below = 0
        last_event = -1e9
        high = low = None
        next_bin = DETECT_WINDOW_SAMPLES
        stop_requested = False
        while not stop_requested:
            await asyncio.sleep(0.025)
            rows = RECORDER.rows
            if len(rows) > seen_rows:
                processed.extend((r[2], r[1]) for r in rows[seen_rows:])
                seen_rows = len(rows)
                last_packet_at = time.perf_counter()
            elapsed = time.perf_counter() - origin
            # Fast 20-sample mean-centered RMS, emitted every 10 samples.
            while next_bin <= len(processed):
                chunk = [float(x[0]) for x in processed[
                    next_bin - DETECT_WINDOW_SAMPLES:next_bin]]
                mean = sum(chunk) / len(chunk)
                rms = math.sqrt(sum((x - mean) ** 2 for x in chunk) / len(chunk))
                t = next_bin / recorder_module.SAMPLING_RATE
                points.append({"t": round(t, 3), "v": rms})
                if len(points) > 800:
                    points = points[-800:]
                if high is None:
                    # Establish threshold from the initial two seconds only.
                    if t >= 2.0:
                        baseline = [p["v"] for p in points if p["t"] <= 2.0]
                        high, low = robust_threshold(baseline)
                        update(phase="LIVE", message="Fast detector live. One threshold crossing triggers; release to re-arm.", high=high, low=low)
                else:
                    if rms >= high:
                        above += 1; below = 0
                        if armed and above >= 1 and t - last_event >= 0.25:
                            event_no += 1
                            last_event = t
                            armed = False
                            with STATE_LOCK:
                                STATE["events"].append({"id": event_no, "t": round(t, 3)})
                                STATE["message"] = f"Detected clench {event_no}. Release to re-arm."
                    else:
                        above = 0
                        if rms < low: below += 1
                        else: below = 0
                        if below >= 3: armed = True
                next_bin += DETECT_HOP_SAMPLES
            update(samples=len(RECORDER.rows), elapsed_s=round(elapsed, 2), envelope=points)
            if time.perf_counter() - last_packet_at > 3.0:
                update(phase="ERROR", error="EMG packet timeout after 3 seconds",
                       message="No EMG packets for 3 seconds. Stream stopped; saving the partial take.")
                stop_requested = True
            else:
                stop_requested = STOP_REQUEST.is_set()
        if snapshot()["phase"] != "ERROR":
            update(phase="STOPPING", message="Stopping stream and saving this take…")
    except Exception as exc:
        update(phase="ERROR", error=str(exc), message=f"Live connection failed: {exc}")
    finally:
        if RECORDER is not None:
            try:
                if RECORDER.client and RECORDER.client.is_connected:
                    await RECORDER.stop_and_disconnect()
                elif not RECORDER.raw_file.closed:
                    RECORDER.raw_file.close()
                adc_path = RECORDER.save_raw_adc() if RECORDER.raw_adc_rows else None
                csv_path = None
                if RECORDER.rows:
                    csv_path = RECORDER.save_filtered()
                phase = "DONE" if snapshot()["phase"] != "ERROR" else "ERROR"
                update(phase=phase, message=(f"Stopped. Saved {len(RECORDER.rows):,} filtered samples; raw ADC CSV and packet log preserved." if phase == "DONE" else snapshot()["message"]),
                       csv_path=str(csv_path) if csv_path else None,
                       raw_adc_path=adc_path, raw_path=RECORDER.raw_path)
            except Exception as save_exc:
                update(phase="ERROR", error=f"Save/cleanup: {save_exc}", message=f"Save/cleanup error: {save_exc}")
        RECORDER = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(GAME_DIR), **kwargs)

    def log_message(self, fmt, *args):
        print("[http] " + fmt % args)

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            return self._json(snapshot())
        return super().do_GET()

    def do_POST(self):
        global START_REQUEST
        path = urlparse(self.path).path
        if path == "/api/start":
            state = snapshot()
            if state["phase"] not in ("READY", "DONE", "ERROR"):
                return self._json({"error": "A live session is already active."}, 409)
            STOP_REQUEST.clear()
            LOOP.call_soon_threadsafe(START_REQUEST.set)
            return self._json({"ok": True})
        if path == "/api/stop":
            STOP_REQUEST.set()
            return self._json({"ok": True})
        return self._json({"error": "Not found"}, 404)


async def main():
    global LOOP, START_REQUEST, STOP_REQUEST
    LOOP = asyncio.get_running_loop()
    START_REQUEST = asyncio.Event()
    STOP_REQUEST = threading.Event()
    server = ThreadingHTTPServer(("127.0.0.1", 8767), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("Live flap game: http://127.0.0.1:8767/flap_replay.html")
    try:
        while True:
            await START_REQUEST.wait()
            START_REQUEST.clear()
            await live_run()
    finally:
        server.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
