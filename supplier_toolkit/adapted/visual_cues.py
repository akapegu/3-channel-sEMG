"""Visual-only supplier trial cues, preserving host dispatch times and repetitions."""
import asyncio
import json
import random
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

INSTRUCTIONS={
 'rest':'NO COMMAND: stay relaxed; only move naturally, and do not perform pinch, flexion, or extension.',
 'pinch':'Gently touch thumb to index fingertip; other fingers loose; wrist steady.',
 'flexion':'Gently bend your hand toward the palm side; forearm supported; fingers loose.',
 'extension':'Gently bend your hand toward the back-of-hand side; forearm supported; fingers loose.'}
ORIENTATION_TEXT={
 'neutral':'thumb-up / neutral forearm (wrist straight)',
 'palm_up':'palm facing upward (forearm supported)',
 'palm_down':'palm facing downward (forearm supported)'}
PAGE='''<!doctype html><meta charset="utf-8"><title>Gesture trial timer</title>
<style>body{font:23px system-ui;background:#111827;color:#f3f4f6;text-align:center;padding:24px}h1{font-size:50px;margin:20px}#timer{font-size:80px}p{max-width:780px;margin:20px auto}button{font:inherit;padding:14px 30px;margin:8px}#detail{min-height:65px}</style>
<h1 id="phase">READY</h1><div id="timer">—</div><p id="detail">Connecting…</p><p id="trial"></p><button id="start">Start trial</button><button id="stop">Stop trial</button><p id="status"></p>
<script>
const start=document.getElementById('start'),stop=document.getElementById('stop');
start.onclick=async()=>{await fetch('/start',{method:'POST'});start.disabled=true;};stop.onclick=async()=>{await fetch('/stop',{method:'POST'});stop.disabled=true;};
async function update(){try{const response=await fetch('/status',{cache:'no-store'});const s=await response.json();document.getElementById('phase').textContent=s.phase;document.getElementById('timer').textContent=s.seconds_left.toFixed(1);document.getElementById('detail').textContent=s.instruction;document.getElementById('trial').textContent='Trial remaining: '+s.trial_seconds_left.toFixed(1)+' s'+(s.repetition?' · Repetition '+s.repetition:'');document.getElementById('status').textContent=s.status;start.hidden=s.phase!=='READY';if(s.done){stop.hidden=true;return;}}catch(e){document.getElementById('status').textContent='Recorder unavailable — relax';}setTimeout(update,80);}update();
</script>'''


def build_schedule(gesture,seconds,hold_min,hold_max,gap,jitter,seed,orientation_sequence=None):
    rng=random.Random(seed);schedule=[];elapsed=0;rep=0
    orientation_sequence=orientation_sequence or ['neutral']
    if gesture=='rest':
        transition=4.0 if len(orientation_sequence)>1 else 0.0
        hold=(seconds-transition*(len(orientation_sequence)-1))/len(orientation_sequence)
        for i,orientation in enumerate(orientation_sequence):
            schedule.append(('REST',hold,i+1,orientation));elapsed+=hold
            if i+1<len(orientation_sequence):
                schedule.append(('RELAX',transition,i+1,None));elapsed+=transition
        return schedule
    while elapsed<seconds:
        relax=max(.3,gap+rng.uniform(-jitter,jitter))
        if elapsed+relax+hold_max+.2>seconds:
            schedule.append(('RELAX',seconds-elapsed,rep,None));break
        next_orientation=orientation_sequence[rep%len(orientation_sequence)]
        schedule.append(('RELAX',relax,rep,next_orientation));elapsed+=relax
        rep+=1;hold=rng.uniform(hold_min,hold_max)
        orientation=orientation_sequence[(rep-1)%len(orientation_sequence)]
        schedule.append((gesture.upper(),hold,rep,orientation));elapsed+=hold
    return schedule


async def run_visual_trial(gesture,recorders,seconds,hold_min,hold_max,gap,jitter,path,orientation_sequence=None):
    requested=threading.Event();aborted=threading.Event()
    orientation_sequence=orientation_sequence or ['neutral']
    if any(x not in ORIENTATION_TEXT for x in orientation_sequence):
        raise ValueError(f'Unknown forearm orientation in {orientation_sequence}')
    first_orientation=orientation_sequence[0]
    ready_instruction=(INSTRUCTIONS[gesture]+f" First posture: {ORIENTATION_TEXT[first_orientation]}.")
    state={'phase':'READY','seconds_left':seconds,'trial_seconds_left':seconds,'repetition':0,
           'instruction':ready_instruction,'status':'Connected · visual cues only · click Start when settled','done':False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            payload=json.dumps(state).encode() if self.path=='/status' else PAGE.encode()
            self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/status' else 'text/html; charset=utf-8');self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
        def do_POST(self):
            host=self.headers.get('Host','')
            if host!=f'127.0.0.1:{server.server_port}' or self.headers.get('Origin')!=f'http://{host}' or self.path not in ('/start','/stop'):
                self.send_error(403);return
            (requested if self.path=='/start' else aborted).set();self.send_response(204);self.end_headers()
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'VISUAL CUE PAGE: http://127.0.0.1:{server.server_port}',flush=True)
    seed=secrets.randbits(32);schedule=build_schedule(gesture,seconds,hold_min,hold_max,gap,jitter,seed,orientation_sequence)
    reps=[];events=[];status='interrupted';protocol_start=None
    def event(phase,rep,orientation=None):
        now=time.time();events.append({'phase':phase,'repetition':rep,'orientation':orientation,'host_unix_ns':time.time_ns(),'host_monotonic_ns':time.perf_counter_ns(),'received_samples':[len(r.rows) for r in recorders]});return now
    async def phase_wait(phase,duration,rep,deadline):
        if phase in (gesture.upper(),'REST'):
            orientation=orientation_sequence[(max(1,rep)-1)%len(orientation_sequence)]
            if phase=='REST':
                instruction=INSTRUCTIONS['rest']+f" Posture: {ORIENTATION_TEXT[orientation]}. Gentle natural non-target movement is okay."
            else:
                instruction=INSTRUCTIONS[gesture]+f" Posture: {ORIENTATION_TEXT[orientation]}. Hold this posture during the movement."
        else:
            orientation=orientation_sequence[min(rep,len(orientation_sequence)-1)]
            instruction=f"Relax the hand; prepare the next posture: {ORIENTATION_TEXT[orientation]}. Do not perform a target gesture."
        state.update(phase=phase,repetition=rep,instruction=instruction,status='Follow the on-screen instruction')
        while time.perf_counter()<deadline:
            if aborted.is_set():raise InterruptedError('User stopped visual trial')
            if not all(r.client.is_connected for r in recorders):raise ConnectionError('Band disconnected')
            state['seconds_left']=max(0,deadline-time.perf_counter());state['trial_seconds_left']=max(0,seconds-(time.perf_counter()-protocol_start)) if protocol_start else seconds
            await asyncio.sleep(.02)
    try:
        wait_until=time.perf_counter()+300
        while not requested.is_set():
            if aborted.is_set():raise InterruptedError('User stopped before Start')
            if time.perf_counter()>wait_until:raise TimeoutError('Start timed out')
            if not all(r.client.is_connected for r in recorders):raise ConnectionError('Band disconnected')
            await asyncio.sleep(.05)
        event('GET READY',0);await phase_wait('GET READY',5,0,time.perf_counter()+5)
        protocol_start=time.perf_counter();elapsed=0
        for phase,duration,rep,orientation in schedule:
            stamp=event(phase,rep,orientation)
            if phase==gesture.upper() or phase=='REST':
                reps.append({'go':stamp,'stop':None,'hold_sec':duration,'orientation':orientation})
            await phase_wait(phase,duration,rep,protocol_start+elapsed+duration)
            elapsed+=duration
            if phase==gesture.upper():reps[-1]['stop']=time.time()
        event('DONE',0);status='complete'
    except InterruptedError:
        status='user_stopped'
    finally:
        for recorder in recorders:
            recorder.visual_cue_status=status
        state.update(phase='DONE — RELAX' if status=='complete' else 'STOPPED — RELAX',seconds_left=0,trial_seconds_left=0,done=True,status='Trial finished; saving recording')
        path.write_text(json.dumps({'status':status,'gesture':gesture,'seconds':seconds,'seed':seed,
            'orientation_sequence':orientation_sequence,'schedule':schedule,'repetitions':reps,'events':events,
            'note':'Host cue dispatch times; browser polling/rendering and human reaction delays remain. No sound cues. Original packets saved separately.'},indent=2))
        await asyncio.sleep(2)
        server.shutdown();server.server_close()
    return reps
