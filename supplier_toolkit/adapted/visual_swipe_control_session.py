"""Visual, randomized four-swipe plus tilt-only negative-control session."""
import asyncio
import json
import random
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


PAGE = '''<!doctype html><meta charset="utf-8"><title>Swipe EMG re-wear session</title>
<style>
body{font:22px system-ui;background:#111827;color:#f3f4f6;text-align:center;padding:22px}
h1{font-size:100px;line-height:1.05;margin:12px}#arrow{font-size:88px;line-height:1.0}#timer{font-size:76px}p{max-width:960px;margin:14px auto}
button{font:inherit;padding:14px 30px;margin:8px}
.small{font-size:17px;color:#cbd5e1}
</style>
<h1 id="cue">READY</h1><div id="arrow"></div><div id="timer">—</div>
<p id="progress"></p><button id="start">Start continuous recording</button>
<button id="stop" hidden>Stop and save partial recording</button><p id="status"></p>
<p class="small">GESTURE: your saved movement · TILT + arrow: fingers relaxed, tilt only · REST: relax</p>
<script>
const start=document.getElementById('start'),stop=document.getElementById('stop');
start.onclick=async()=>{start.disabled=true;await fetch('/start',{method:'POST'});};
stop.onclick=async()=>{stop.disabled=true;await fetch('/stop',{method:'POST'});};
async function update(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();
document.getElementById('cue').textContent=s.cue_word;
document.getElementById('arrow').textContent=s.cue_arrow||'';
document.getElementById('timer').textContent=s.seconds_left.toFixed(1);
document.getElementById('progress').textContent=s.progress;
document.getElementById('status').textContent=s.status;
start.hidden=s.phase!=='READY';stop.hidden=s.phase==='READY'||s.done;
if(s.done)return;}catch(e){document.getElementById('status').textContent='Recorder unavailable — relax.';}
setTimeout(update,100);}update();
</script>'''


def make_schedule(plan):
    rng = random.Random(secrets.randbits(32))
    trials = []
    for direction in plan["directions"]:
        for repetition in range(1, int(plan["repetitions_per_gesture"]) + 1):
            trials.append({"trial_type": "gesture", "label": direction,
                           "target_gesture": direction, "direction": direction,
                           "repetition": repetition})
        for repetition in range(1, int(plan["tilt_only_repetitions_per_direction"]) + 1):
            trials.append({"trial_type": "tilt_only_control", "label": "rest",
                           "target_gesture": None, "direction": direction,
                           "repetition": repetition})
    rng.shuffle(trials)

    schedule = []
    for index, trial in enumerate(trials, start=1):
        if index > 1 and (index - 1) % int(plan["break_every_trials"]) == 0:
            schedule.append({"phase": "BREAK", "duration": float(plan["break_seconds"]),
                             "label": "rest", "trial_index": index})
        common = {**trial, "trial_index": index,
                  "acquisition_group": f"{plan['acquisition_id']}_{plan['session']}",
                  "partition": "development"}
        if trial["trial_type"] == "gesture":
            action = (f"GESTURE {trial['direction'].upper()} · repetition {trial['repetition']} of "
                      f"{plan['repetitions_per_gesture']}. {plan['gesture_instruction']}")
        else:
            action = (f"TILT ONLY {trial['direction'].upper()} · control {trial['repetition']} of "
                      f"{plan['tilt_only_repetitions_per_direction']}. "
                      f"{plan['tilt_control_instruction']}")
        schedule.append({**common, "phase": "RELAX", "duration": float(plan["pre_cue_rest_seconds"]),
                         "label": "rest", "instruction": "RELAX. Release your fingers and wait for the next cue."})
        schedule.append({**common, "phase": "ACTION", "duration": float(plan["active_cue_seconds"]),
                         "instruction": action})
    schedule.append({"phase": "FINAL REST", "duration": float(plan["pre_cue_rest_seconds"]),
                     "label": "rest", "instruction": "Relax completely. The recording is ending."})
    for phase_index, entry in enumerate(schedule):
        entry["phase_index"] = phase_index
    total = float(plan["initial_settle_seconds"]) + sum(x["duration"] for x in schedule)
    return schedule, round(total)


ARROWS = {"down": "↓", "top": "↑", "left": "←", "right": "→"}


def visible_cue(entry):
    if entry["phase"] != "ACTION":
        return "REST", ""
    if entry.get("trial_type") == "tilt_only_control":
        return "TILT", ARROWS[entry["direction"]]
    return entry["direction"].upper(), ""


async def run_session(recorders, cue_path, plan_path, connect_and_start):
    plan = json.loads(plan_path.read_text())
    schedule, total = make_schedule(plan)
    requested, stopped = threading.Event(), threading.Event()
    session_number = "".join(ch for ch in str(plan["session"]) if ch.isdigit()) or "?"
    session_label = f"Session {session_number}"
    n_trials = int(plan["repetitions_per_gesture"]) * len(plan["directions"])
    n_trials += int(plan["tilt_only_repetitions_per_direction"]) * len(plan["directions"])
    state = {"phase": "READY", "seconds_left": 0.0, "cue_word": "READY", "cue_arrow": "",
             "progress": f"{session_label.upper()} / 4 · {n_trials} ACTIONS · {total / 60:.1f} MIN",
             "status": "Band found · not connected or recording yet", "done": False}

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
    print(f"VISUAL SWIPE CONTROL PAGE: http://127.0.0.1:{server.server_port}/", flush=True)
    events, status, session_start = [], "interrupted", None

    def emit(name, entry=None):
        event = {"event": name, "host_unix_ns": time.time_ns(),
                 "host_monotonic_ns": time.perf_counter_ns(),
                 "sample_index": len(recorders[0].rows)}
        if entry:
            for key in ("partition", "acquisition_group", "phase", "phase_index", "label",
                        "trial_type", "trial_index", "target_gesture", "direction", "repetition"):
                if key in entry:
                    event[key] = entry[key]
        events.append(event)
        return event

    def present_phase(entry, progress):
        cue_word, cue_arrow = visible_cue(entry)
        state.update(phase=entry["phase"], seconds_left=entry["duration"],
                     cue_word=cue_word, cue_arrow=cue_arrow, progress=progress,
                     status="BLE capture running · follow the visible cue")

    async def wait_phase(entry, deadline, progress):
        present_phase(entry, progress)
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
        state.update(phase="CONNECTING", cue_word="WAIT", cue_arrow="",
                     status="Start received · connecting over BLE")
        await connect_and_start()
        session_start = emit("session_start")
        started = time.perf_counter()
        settle = {"phase": "SETTLE", "duration": float(plan["initial_settle_seconds"]),
                  "label": "rest", "instruction": "Stay relaxed and still while recording settles."}
        present_phase(settle, f"{session_label} · initial relaxed baseline")
        emit("label_start", settle)
        emit("cue_start", settle)
        await wait_phase(settle, started + settle["duration"], f"{session_label} · initial relaxed baseline")
        emit("cue_stop", settle)
        emit("label_stop", settle)
        elapsed = settle["duration"]
        trial_index = 0
        for entry in schedule:
            if entry.get("trial_index"):
                trial_index = entry["trial_index"]
            if entry["phase"] in ("BREAK", "FINAL REST"):
                pass
            present_phase(entry, f"{trial_index} / {n_trials}")
            emit("label_start", entry)
            emit("cue_start", entry)
            await wait_phase(entry, started + elapsed + entry["duration"],
                             f"{trial_index} / {n_trials}")
            elapsed += entry["duration"]
            emit("cue_stop", entry)
            emit("label_stop", entry)
        emit("session_stop")
        status = "complete"
    except InterruptedError:
        status = "user_stopped"
    finally:
        for recorder in recorders:
            recorder.visual_cue_status = status
        state.update(phase="DONE" if status == "complete" else "STOPPED",
                     cue_word="DONE" if status == "complete" else "STOP",
                     cue_arrow="", seconds_left=0, done=True,
                     status=f"Session {status} · preserving received data")
        cue_path.write_text(json.dumps({
            "status": status, "sampling_rate": plan["sampling_rate_hz"],
            "preprocessing_profile": plan["preprocessing_profile"],
            "acquisition_id": plan["acquisition_id"], "session": plan["session"],
            "plan": plan, "planned_duration_seconds": total,
            "session_start_sample": session_start["sample_index"] if session_start else None,
            "schedule": schedule, "events": events,
            "labeling_note": "Gesture trials are labeled by intended gesture. Tilt-only trials are labeled rest/no-command and retain direction metadata.",
            "timing_note": "Host cue dispatch timestamps and sample indices mark the visual cue interval; they include user response timing.",
            "split_note": "This is one development session, not an independent test. Keep the entire session group together in later splits."
        }, indent=2) + "\n")
        await asyncio.sleep(1)
        server.shutdown()
        server.server_close()
    return {"status": status, "cue_file": str(cue_path), "event_count": len(events),
            "planned_duration_seconds": total, "planned_action_trials": n_trials}
