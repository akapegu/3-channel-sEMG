"""Compare raw descriptive features across two sequential cued acquisitions.
No trained classifier, resampling, threshold fitting or acquisition-time claim.
"""
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out={}
for name,stem,action in [('clench','exp004_attempt04_raw','GENTLE CLENCH'),('tilt','exp004_motioncontrol01_raw','GENTLE FOREARM TILT')]:
    path=ROOT/'recordings'/f'{stem}.csv'
    events=json.loads(path.with_suffix('.cues.json').read_text())['events']
    origin=next(e['host_monotonic_ns'] for e in events if e['action']=='RELAX')
    with path.open() as f:r=list(csv.DictReader(f))
    t=np.array([(int(x['host_monotonic_ns'])-origin)/1e9 for x in r])
    x=np.array([[int(y[f'ch{c}_adc']) for c in (1,2,3)] for y in r])
    with path.with_suffix('.imu.csv').open() as f:r=list(csv.DictReader(f))
    mt=np.array([(int(y['host_monotonic_ns'])-origin)/1e9 for y in r])
    mx=np.array([[int(y[f'accel_{c}_counts']) for c in 'xyz'] for y in r])
    phases={'initial_rest':[(2,9)],'hold':[],'release':[],'late_recovery':[],'final_rest':[(62,69)]}
    for i,e in enumerate(events):
        if e['action']==action:
            a=(e['host_monotonic_ns']-origin)/1e9;b=(events[i+1]['host_monotonic_ns']-origin)/1e9
            phases['hold'].append((a+1,b-.5));phases['release'].append((b+.25,b+1.5));phases['late_recovery'].append((b+3,b+5.5))
    base=np.median(mx[(mt>=2)&(mt<9)],axis=0)
    result={}
    for phase,spans in phases.items():
        rms=[];medians=[]
        for a,b in spans:
            v=x[(t>=a)&(t<b)]
            for j in range(0,len(v)-249,250):
                z=v[j:j+250];rms.append(np.sqrt(np.mean((z-z.mean(axis=0))**2,axis=0)))
            medians.append(np.median(mx[(mt>=a)&(mt<b)],axis=0))
        result[phase]={'median_centered_rms_counts':np.median(rms,axis=0).tolist(),
                       'median_distance_of_interval_accel_median_from_initial_rest_counts':float(np.median(np.linalg.norm(np.array(medians)-base,axis=1)))}
    out[name]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'phases':result}
out['limitations']=['Separate sequential acquisitions, not randomised or independent held-out evaluation.',
                    'Motion-only means no intentional clenching; postural muscle activity can remain.',
                    'Fixed cue-relative margins; host arrival timing retains BLE/display/reaction uncertainty.',
                    'Acceleration median distance is descriptive and includes gravity orientation; not displacement or calibrated acceleration.',
                    'No classifier, significance, physiological-source or SNR claim.']
p=ROOT/'analysis/clench_motion_comparison';p.mkdir(exist_ok=True)
(p/'metrics.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
