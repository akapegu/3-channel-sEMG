"""Start-gated visual cues and sample-indexed events for the swipe4 capture."""
import asyncio
import json
import random
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GESTURES = ("down", "left", "right", "top")
PAGE = '''<!doctype html><meta charset="utf-8"><title>Four-swipe EMG session</title>
<style>
body{font:22px system-ui;background:#111827;color:#f3f4f6;text-align:center;padding:22px}
h1{font-size:46px;margin:12px}#timer{font-size:76px}p{max-width:900px;margin:14px auto}
button{font:inherit;padding:14px 30px;margin:8px}#detail{min-height:75px}
.small{font-size:17px;color:#cbd5e1}
</style>
<h1 id="phase">READY</h1><div id="timer">—</div><p id="detail">Loading plan…</p>
<p id="progress"></p><button id="start">Start one continuous recording</button>
<button id="stop" hidden>Stop and save partial recording</button><p id="status"></p>
<p class="small">Use the same movement you used for the matching swipe CSV. Move once during the direction cue; release fully when REST appears. Keep your band position steady.</p>
<script>
const start=document.getElementById('start'),stop=document.getElementById('stop');
start.onclick=async()=>{start.disabled=true;await fetch('/start',{method:'POST'});};
stop.onclick=async()=>{stop.disabled=true;await fetch('/stop',{method:'POST'});};
async function update(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();
document.getElementById('phase').textContent=s.phase;
document.getElementById('timer').textContent=s.seconds_left.toFixed(1);
document.getElementById('detail').textContent=s.instruction;
document.getElementById('progress').textContent=s.progress;
document.getElementById('status').textContent=s.status;
start.hidden=s.phase!=='READY';stop.hidden=s.phase==='READY'||s.done;
if(s.done)return;}catch(e){document.getElementById('status').textContent='Recorder unavailable — relax.';}
setTimeout(update,100);}update();
</script>'''


def make_schedule(plan):
    schedule = []
    reps_per_class = int(plan["repetitions_per_class_per_partition"])
    if reps_per_class <= 0:
        raise ValueError("repetitions_per_class_per_partition must be positive")
    task_id = 0
    seed = secrets.randbits(32)
    rng = random.Random(seed)
    for block_index, partition in enumerate(plan["partitions"], 1):
        schedule.append({"phase": "BLOCK REST", "duration": int(plan["block_transition_relax_seconds"]),
                         "label": "rest", "partition": partition,
                         "group": f"{plan['acquisition_id']}_{partition}",
                         "instruction": f"Stay relaxed. The {partition.upper()} block starts after this pause."})
        trials = [gesture for gesture in GESTURES for _ in range(reps_per_class)]
        rng.shuffle(trials)
        repetition_counts = {gesture: 0 for gesture in GESTURES}
        for gesture in trials:
            repetition_counts[gesture] += 1
            task_id += 1
            base = {"partition": partition,
                    "group": f"{plan['acquisition_id']}_{partition}",
                    "gesture": gesture, "task_id": task_id,
                    "repetition": repetition_counts[gesture],
                    "block_index": block_index, "schedule_seed": seed}
            schedule.append({**base, "phase": "REST", "duration": float(plan["relax_before_each_cue_seconds"]),
                             "label": "rest",
                             "instruction": f"RELAX. Release completely. Next movement: {gesture.upper()}."})
            schedule.append({**base, "phase": "SWIPE", "duration": float(plan["active_cue_seconds"]),
                             "label": gesture,
                             "instruction": f"SWIPE {gesture.upper()} NOW — perform your familiar {gesture} movement once."})
    schedule.append({"phase": "FINAL REST", "duration": float(plan["relax_before_each_cue_seconds"]),
                     "label": "rest", "partition": plan["partitions"][-1],
                     "group": f"{plan['acquisition_id']}_{plan['partitions'][-1]}",
                     "instruction": "Relax completely. The recording is ending."})
    for index, entry in enumerate(schedule):
        entry["phase_index"] = index
    return schedule, round(float(plan["initial_settle_seconds"]) + sum(x["duration"] for x in schedule))


async def run_session(recorders, cue_path, plan_path, connect_and_start):
    plan = json.loads(plan_path.read_text())
    schedule, total = make_schedule(plan)
    requested, stopped = threading.Event(), threading.Event()
    estimated = total / 60
    state = {"phase": "READY", "seconds_left": 0.0,
             "instruction": (f"One continuous recording, about {estimated:.1f} minutes. "
                             "Each direction appears equally often in randomized order in TRAIN, VALIDATION, and TEST blocks. "
                             "Use the exact movement from its supplied CSV; keep a comfortable stable arm and band position."),
             "progress": "8 repetitions per direction per block · 24 prompted swipes per direction total",
             "status": "Band found · not connected or recording yet · settle before Start",
             "done": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            payload = json.dumps(state).encode() if self.path == "/status" else PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if self.path == "/status" else "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            host = self.headers.get("Host", "")
            if (host != f"127.0.0.1:{server.server_port}" or
                self.headers.get("Origin") != f"http://{host}" or
                self.path not in ("/start", "/stop")):
                self.send_error(403)
                return
            (requested if self.path == "/start" else stopped).set()
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"VISUAL SWIPE SESSION PAGE: http://127.0.0.1:{server.server_port}/", flush=True)
    events = []
    status = "interrupted"
    started = None

    def emit(name, entry=None):
        event = {"event": name, "host_unix_ns": time.time_ns(),
                 "host_monotonic_ns": time.perf_counter_ns(),
                 "sample_index": len(recorders[0].rows)}
        if entry:
            for key in ("partition", "group", "gesture", "task_id", "repetition",
                        "block_index", "phase_index", "label"):
                event[key if key != "group" else "acquisition_group"] = entry.get(key)
            event["actual_orientation_verified"] = False
        events.append(event)
        return event

    async def wait_phase(entry, deadline):
        state.update(phase=entry["phase"], seconds_left=entry["duration"],
                     instruction=entry["instruction"],
                     progress=f"{entry.get('partition', '').upper()} · {entry.get('gesture', 'relax').upper()}"
                              f" · repetition {entry.get('repetition', '—')} / {plan['repetitions_per_class_per_partition']} per direction",
                     status="BLE capture is running · follow only the visible cue")
        while time.perf_counter() < deadline:
            if stopped.is_set():
                raise InterruptedError("User stopped the continuous session")
            if not all(r.client and r.client.is_connected for r in recorders):
                raise ConnectionError("Band disconnected during the session")
            state["seconds_left"] = max(0.0, deadline - time.perf_counter())
            await asyncio.sleep(.02)

    try:
        wait_until = time.perf_counter() + 600
        while not requested.is_set():
            if stopped.is_set():
                raise InterruptedError("User stopped before Start")
            if time.perf_counter() > wait_until:
                raise TimeoutError("Start was not pressed within 10 minutes")
            await asyncio.sleep(.05)
        state.update(phase="CONNECTING", instruction="Connecting to the band, then starting the one continuous recording.",
                     status="Start received · connecting over BLE")
        await connect_and_start()
        session_start = emit("session_start")
        started = time.perf_counter()
        state.update(phase="SESSION SETTLE", seconds_left=float(plan["initial_settle_seconds"]),
                     instruction="Stay relaxed and still while the recording settles.",
                     progress="The full source recording has started.", status="BLE connected · recording")
        await asyncio.sleep(float(plan["initial_settle_seconds"]))
        emit("session_settle_stop")
        elapsed = float(plan["initial_settle_seconds"])
        for entry in schedule:
            if entry["phase"] == "BLOCK REST":
                emit("block_start", entry)
            emit("label_start", entry)
            emit("cue_start", entry)
            await wait_phase(entry, started + elapsed + entry["duration"])
            elapsed += entry["duration"]
            emit("cue_stop", entry)
            emit("label_stop", entry)
            if entry["phase"] == "BLOCK REST":
                emit("block_stop", entry)
        emit("session_stop")
        status = "complete"
    except InterruptedError:
        status = "user_stopped"
    finally:
        for recorder in recorders:
            recorder.visual_cue_status = status
        state.update(phase="DONE — RELAX" if status == "complete" else "STOPPED — RELAX",
                     seconds_left=0, done=True,
                     instruction="Relax. Saving the original continuous recording and cue timeline.",
                     status=f"Session {status} · preserving received data")
        cue_path.write_text(json.dumps({
            "status": status, "sampling_rate": 1000,
            "preprocessing_profile": "npg-native1000-bs48_52-hp70-v1",
            "acquisition_id": plan["acquisition_id"], "plan": plan,
            "planned_duration_seconds": total, "session_start_sample": session_start["sample_index"] if events else None,
            "schedule": schedule, "events": events,
            "labeling_note": "REST is the no-target class. Swipe label intervals begin at the visual cue dispatch and end at cue stop; cue onset includes human reaction time. Review the cues before training.",
            "timing_note": "Host dispatch timestamps/sample indices approximate visual presentation and physiology latency.",
            "split_note": plan["partition_note"]
        }, indent=2) + "\n")
        await asyncio.sleep(2)
        server.shutdown()
        server.server_close()
    return {"status": status, "cue_file": str(cue_path), "event_count": len(events),
            "planned_duration_seconds": total}
