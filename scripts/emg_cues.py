"""Local browser cues; no network upload. Cue timestamps are host scheduling times."""
import asyncio
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def protocol(clench_seconds=2, relax_seconds=3, action='GENTLE CLENCH'):
    phases = [('RELAX', 10, 0)]
    if action=='MIXED':
        rng=random.Random(20261101)
        for repetition in range(1,5):
            actions=['GENTLE CLENCH','GENTLE FOREARM TILT']
            rng.shuffle(actions)
            for movement in actions:
                phases.extend([(movement,clench_seconds,repetition),('RELAX',relax_seconds,repetition)])
        return phases+[('RELAX',10,0)]
    for repetition in range(1, 6):
        phases.extend([(action, clench_seconds, repetition), ('RELAX', relax_seconds, repetition)])
    return phases + [('RELAX', 10, 0)]


PAGE = '''<!doctype html><meta charset="utf-8"><title>EMG timing</title>
<style>body{font:24px system-ui;text-align:center;background:#101820;color:#f4f6f7;padding:35px}h1{font-size:48px}#remaining{font-size:80px}button{font:inherit;padding:16px 36px}p{max-width:750px;margin:20px auto}</style>
<h1 id="action">Ready for experiment</h1><div id="remaining">Waiting for recorder</div>
<p id="detail">Follow this run's movement instructions.</p>
<button id="start">Start experiment</button><p id="connection">Connecting to local recorder…</p>
<script>
const button=document.getElementById('start');
button.onclick=async()=>{button.disabled=true;try{const r=await fetch('/start',{method:'POST'});if(!r.ok)throw Error('Start failed');}catch(e){button.disabled=false;document.getElementById('connection').textContent=e.message;}};
async function update(){try{const r=await fetch('/status',{cache:'no-store'});const s=await r.json();document.getElementById('action').textContent=s.action;document.getElementById('remaining').textContent=s.remaining.toFixed(1);document.getElementById('detail').textContent=s.detail;document.getElementById('connection').textContent='Local recorder connected';if(s.action!=='Ready for experiment')button.hidden=true;if(s.action==='DONE — RELAX')return;}catch(e){document.getElementById('connection').textContent='Recorder unavailable — relax';}setTimeout(update,80);}update();
</script>'''


async def run_cues(client, rows, path, clench_seconds=2, relax_seconds=3, action='GENTLE CLENCH'):
    requested = threading.Event()
    schedule = protocol(clench_seconds, relax_seconds, action)
    protocol_type=action
    repetitions=max(p[2] for p in schedule)
    total = sum(p[1] for p in schedule)
    state = {'action': 'Ready for experiment', 'remaining': total,
             'detail': 'Click Start when you can see this page. Five-second preparation follows.'}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            body = json.dumps(state).encode() if self.path == '/status' else PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if self.path == '/status' else 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            # Same-origin local control; no cross-origin POST accepted.
            host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            if self.path != '/start' or host != f'127.0.0.1:{server.server_port}' or origin != f'http://{host}':
                self.send_error(403)
                return
            requested.set()
            self.send_response(204)
            self.end_headers()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'CUE PAGE: {url}', flush=True)
    path.with_suffix('.url.txt').write_text(url)
    events = []
    status = 'interrupted'
    try:
        # Buffer stays bounded: stop if page is not started within five minutes.
        deadline = time.perf_counter() + 300
        while not requested.is_set():
            if not client.is_connected or time.perf_counter() > deadline:
                raise RuntimeError('Band disconnected or cue Start timed out')
            await asyncio.sleep(.05)
        phases = [('GET READY — RELAX', 5, 0)] + schedule
        start = time.perf_counter()
        elapsed = 0
        for action, duration, repetition in phases:
            detail=f'Repetition {repetition}/{repetitions}'
            if action=='GENTLE FOREARM TILT':
                detail+=' — fingers loose; gently tilt forearm and hold'
            elif action=='GENTLE CLENCH':
                detail+=' — gentle fist; forearm supported; no added tilt'
            else:
                detail+=' — release fist / return to supported starting position'
            state.update(action=action, detail=detail if repetition else 'Rest at starting position; hand relaxed')
            events.append({'action': action, 'planned_seconds_from_prepare': elapsed,
                           'repetition': repetition, 'host_monotonic_ns': time.perf_counter_ns(),
                           'host_unix_ns': time.time_ns(), 'received_samples_at_dispatch': len(rows)})
            deadline = start + elapsed + duration
            while time.perf_counter() < deadline:
                if not client.is_connected:
                    raise RuntimeError('Band disconnected during cues')
                state['remaining'] = max(0, deadline-time.perf_counter())
                await asyncio.sleep(.02)
            elapsed += duration
        events.append({'action': 'DONE', 'host_monotonic_ns': time.perf_counter_ns(),
                       'host_unix_ns': time.time_ns(), 'received_samples_at_dispatch': len(rows)})
        state.update(action='DONE — RELAX', remaining=0, detail='Complete. Recording will be saved.')
        status = 'complete'
        await asyncio.sleep(3)
    finally:
        path.write_text(json.dumps({'status': status, 'events': events,
            'protocol_seconds': total, 'clench_seconds': clench_seconds, 'relax_seconds': relax_seconds,
            'protocol_type':protocol_type, 'schedule_seed':20261101 if protocol_type=='MIXED' else None,
            'note': 'Host cue-state changes, not measured screen/audio or muscle onset. '
                    'Browser polls every 80 ms plus rendering/network delay. '
                    'BLE latency remains unknown. Pre-start received samples are retained.'}, indent=2))
        server.shutdown()
        server.server_close()
