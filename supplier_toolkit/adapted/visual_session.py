"""One Start-gated, continuous visual-cue session for all S03 partitions."""
import asyncio
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from visual_cues import INSTRUCTIONS, ORIENTATION_TEXT, build_schedule

SESSION_PAGE='''<!doctype html><meta charset="utf-8"><title>Continuous EMG session</title>
<style>body{font:23px system-ui;background:#111827;color:#f3f4f6;text-align:center;padding:24px}h1{font-size:46px;margin:16px}#timer{font-size:72px}p{max-width:900px;margin:18px auto}button{font:inherit;padding:14px 30px;margin:8px}#detail{min-height:90px}</style>
<h1 id="phase">READY</h1><div id="timer">—</div><p id="detail">Connecting…</p><p id="trial"></p>
<button id="start">Start full session</button><button id="stop">Stop session</button><p id="status"></p>
<script>
const start=document.getElementById('start'),stop=document.getElementById('stop');
start.onclick=async()=>{await fetch('/start',{method:'POST'});start.disabled=true;};
stop.onclick=async()=>{await fetch('/stop',{method:'POST'});stop.disabled=true;};
async function update(){try{const r=await fetch('/status',{cache:'no-store'});const s=await r.json();
document.getElementById('phase').textContent=s.phase;document.getElementById('timer').textContent=s.seconds_left.toFixed(1);
document.getElementById('detail').textContent=s.instruction;document.getElementById('trial').textContent='Full session remaining: '+Math.ceil(s.session_seconds_left/60)+' min';
document.getElementById('status').textContent=s.status;start.hidden=s.phase!=='READY';if(s.done){stop.hidden=true;return;}}
catch(e){document.getElementById('status').textContent='Recorder unavailable — relax';}setTimeout(update,100);}update();
</script>'''

PARTITIONS=('train','validation','test')
GESTURES=('pinch','flexion','extension')
ORIENTATIONS=('neutral','palm_up','palm_down')


def orientation_sequence(partition,gesture):
    shift=(PARTITIONS.index(partition)+GESTURES.index(gesture))%len(ORIENTATIONS)
    return [ORIENTATIONS[(shift+i)%len(ORIENTATIONS)] for i in range(len(ORIENTATIONS))]


def build_session_plan(plan,seconds_per_task=60,prep_seconds=5,transition_seconds=5):
    schedule=[];task_id=0
    acquisition_id=plan.get('acquisition_id','S03')
    partitions=plan.get('partitions',PARTITIONS)
    if not partitions or any(p not in PARTITIONS for p in partitions):
        raise ValueError(f'Invalid partition sequence: {partitions}')
    for partition in partitions:
        group=f"{acquisition_id}_{partition}"
        schedule.append({'phase':'BLOCK','duration':5,'partition':partition,'group':group,
                         'gesture':None,'repetition':0,'orientation':None,
                         'label':'rest',
                         'instruction':f"{partition.upper()} block begins. Stay relaxed; relaxed/no-gesture time is labeled REST."})
        for gesture in plan['partition_orders'][partition]:
            if gesture not in GESTURES:
                raise ValueError(f'Only target gestures belong in partition_orders: {gesture}')
            task_id+=1;seq=orientation_sequence(partition,gesture)
            meta={'partition':partition,'group':group,'gesture':gesture,
                  'trial_in_partition':PARTITIONS.index(partition)+1,
                  'task_id':task_id,'orientation_sequence':seq}
            schedule.append({**meta,'phase':'TASK PREP','duration':prep_seconds,
                             'repetition':0,'orientation':seq[0],
                             'label':'rest',
                             'instruction':f"RELAX — next: {INSTRUCTIONS[gesture]} Optional posture prompt: {ORIENTATION_TEXT[seq[0]]}; use only if comfortable. Wait for the gesture cue."})
            active_seconds=seconds_per_task-prep_seconds
            seed=secrets.randbits(32)
            phases=build_schedule(gesture,active_seconds,3,5,6,.5,seed,seq)
            for phase,duration,rep,orientation in phases:
                if phase=='REST':
                    instruction=INSTRUCTIONS['rest']+f" Optional posture prompt: {ORIENTATION_TEXT[orientation]}. Use it only if comfortable; relaxed natural non-target motion is okay."
                elif phase==gesture.upper():
                    instruction=INSTRUCTIONS[gesture]+f" Optional posture prompt: {ORIENTATION_TEXT[orientation]}; use only if comfortable. The named gesture is the important cue."
                elif orientation:
                    instruction=f"RELAX. Optional posture prompt: {ORIENTATION_TEXT[orientation]}. Change only if comfortable; do not perform a target gesture."
                else:
                    instruction='RELAX. Release the gesture and wait for the next cue.'
                schedule.append({**meta,'phase':phase,'duration':duration,'repetition':rep,
                                 'orientation':orientation,'schedule_seed':seed,
                                 'label':'rest' if phase=='RELAX' else gesture,
                                 'instruction':instruction})
            schedule.append({**meta,'phase':'TASK TRANSITION','duration':transition_seconds,
                             'repetition':0,'orientation':None,
                             'label':'rest',
                             'instruction':'Relax completely. The next labeled task will be shown shortly.'})
    for i,entry in enumerate(schedule):entry['phase_index']=i
    total=sum(x['duration'] for x in schedule)
    return schedule,total


async def run_guided_session(recorders,cue_path,plan_path,seconds_per_task=60,connect_callback=None):
    plan=json.loads(plan_path.read_text())
    schedule,total=build_session_plan(plan,seconds_per_task)
    requested=threading.Event();aborted=threading.Event()
    partition_text=', '.join(p.upper() for p in plan.get('partitions',PARTITIONS))
    evaluation_note=' This run is evaluation-only; no model fitting is performed.' if plan.get('evaluation_only') else ''
    state={'phase':'READY','seconds_left':0,'session_seconds_left':total,
           'instruction':f'One BLE capture will cover {partition_text}. Click Start when settled.{evaluation_note}',
           'status':'Band found · visual cues only · not streaming yet','done':False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            payload=json.dumps(state).encode() if self.path=='/status' else SESSION_PAGE.encode()
            self.send_response(200);self.send_header('Content-Type','application/json' if self.path=='/status' else 'text/html; charset=utf-8')
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
        def do_POST(self):
            host=self.headers.get('Host','')
            if host!=f'127.0.0.1:{server.server_port}' or self.headers.get('Origin')!=f'http://{host}' or self.path not in ('/start','/stop'):
                self.send_error(403);return
            (requested if self.path=='/start' else aborted).set();self.send_response(204);self.end_headers()

    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    print(f'VISUAL SESSION PAGE: http://127.0.0.1:{server.server_port}',flush=True)
    events=[];status='interrupted';started=None

    def event(name,entry=None):
        now=time.time()
        events.append({'event':name,'host_unix_ns':time.time_ns(),
            'host_monotonic_ns':time.perf_counter_ns(),
            'sample_index':len(recorders[0].rows),
            'partition':entry.get('partition') if entry else None,
            'acquisition_group':entry.get('group') if entry else None,
            'gesture':entry.get('gesture') if entry else None,
            'trial_in_partition':entry.get('trial_in_partition') if entry else None,
            'task_id':entry.get('task_id') if entry else None,
            'phase_index':entry.get('phase_index') if entry else None,
            'repetition':entry.get('repetition') if entry else None,
            'orientation_prompt':entry.get('orientation') if entry else None,
            'actual_orientation_verified':False,
            'label':entry.get('label') if entry else None,
            'phase':entry.get('phase') if entry else None})
        return now

    async def wait_phase(entry,deadline):
        phase=entry['phase'];duration=entry['duration'];instruction=entry['instruction']
        state.update(phase=phase.upper(),seconds_left=duration,
                     session_seconds_left=max(0,total-(time.perf_counter()-started)),
                     instruction=instruction,status='Follow each on-screen cue; move only for the named gesture.')
        while time.perf_counter()<deadline:
            if aborted.is_set():raise InterruptedError('User stopped continuous session')
            if not all(r.client.is_connected for r in recorders):raise ConnectionError('Band disconnected')
            state['seconds_left']=max(0,deadline-time.perf_counter())
            state['session_seconds_left']=max(0,total-(time.perf_counter()-started))
            await asyncio.sleep(.02)

    try:
        wait_until=time.perf_counter()+600
        while not requested.is_set():
            if aborted.is_set():raise InterruptedError('User stopped before full-session Start')
            if time.perf_counter()>wait_until:raise TimeoutError('Full-session Start timed out')
            await asyncio.sleep(.05)
        # Do not collect unlabeled idle data while the user reads the page.
        # Connect and start the one continuous BLE stream only after the
        # explicit Start click, then anchor cue sample indices to that stream.
        state.update(phase='CONNECTING',seconds_left=0,
                     instruction='Connecting to the band, then starting the continuous recording.',
                     status='Start received · connecting over BLE')
        if connect_callback is not None:
            await connect_callback()
        await asyncio.gather(*(r.start_streaming() for r in recorders))
        event('session_start')
        start_entry={'phase':'SESSION PREP','duration':5,'instruction':'Session begins in 5 seconds. Keep relaxed; follow the displayed labels.'}
        started=time.perf_counter()
        await wait_phase(start_entry,started+5)
        elapsed=5
        for i,entry in enumerate(schedule):
            if entry['phase']=='TASK PREP':
                event('task_start',entry)
            event('label_start' if entry.get('label') else 'phase_start',entry)
            await wait_phase(entry,started+elapsed+entry['duration'])
            elapsed+=entry['duration']
            if entry.get('label'):
                event('label_stop',entry)
            if entry['phase']=='TASK TRANSITION':
                event('task_stop',entry)
        event('session_stop')
        status='complete'
    except InterruptedError:
        status='user_stopped'
    finally:
        for r in recorders:r.visual_cue_status=status
        state.update(phase='DONE — RELAX' if status=='complete' else 'STOPPED — RELAX',
                     seconds_left=0,session_seconds_left=0,done=True,
                     status='Continuous capture finished; saving original recording')
        cue_path.write_text(json.dumps({'status':status,'sampling_rate':1000,
            'seconds_per_task':seconds_per_task,'planned_duration_seconds':total+5,
            'partitions':plan.get('partitions',PARTITIONS),
            'evaluation_only':bool(plan.get('evaluation_only',False)),
            'acquisition_id':plan.get('acquisition_id'),
            'schedule':schedule,'events':events,
            'orientation_text':ORIENTATION_TEXT,
            'note':'One continuous BLE acquisition. Orientation values are optional prompts, not verified actual postures. Cue/sample indices are host-dispatch markers; reaction delay, browser render latency, and physiology remain approximate. Source packet log is preserved separately.'},indent=2))
        await asyncio.sleep(2);server.shutdown();server.server_close()
    return {'status':status,'cue_path':str(cue_path),'events':events,'schedule':schedule,'planned_duration_seconds':total+5}
