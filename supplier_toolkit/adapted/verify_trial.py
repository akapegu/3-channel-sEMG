"""Read-only acquisition/label QC; writes a separate summary, never changes samples."""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path


def verify(path):
    path=Path(path).resolve()
    processing=json.loads(path.with_suffix('.processing.json').read_text())
    cue=json.loads(path.with_suffix('.beep.json').read_text())
    raw_files=processing['raw_packet_files']
    if len(raw_files)!=1:raise ValueError('Single-band verification only')
    packets=[json.loads(x) for x in Path(raw_files[0]).read_text().splitlines()]
    streams={}
    for kind,name,fmt in [('beb5483e-36e1-4688-b7f5-ea07361b26a8','emg','>BHHH'),('5a153fa9-7be0-400c-8ef8-d84502b31c4d','accel','>Bhhh')]:
        selected=[x for x in packets if x['characteristic']==kind]
        frames=[];valid=[];malformed=empty=unexpected=0
        for packet in selected:
            data=bytes.fromhex(packet['payload_hex'])
            if not data:empty+=1;continue
            if len(data)%7:malformed+=1;continue
            if name=='emg' and len(data)!=140:unexpected+=1
            decoded=list(struct.iter_unpack(fmt,data));frames.extend(decoded);valid.append((packet,len(decoded)))
        if not frames:raise ValueError(f'No {name} frames')
        span=(valid[-1][0]['host_monotonic_ns']-valid[0][0]['host_monotonic_ns'])/1e9
        steps=[(b[0]-a[0])%256 for a,b in zip(frames,frames[1:])]
        streams[name]={'samples':len(frames),'packets':len(selected),'malformed':malformed,'empty':empty,
            'nonunit_transitions':sum(x!=1 for x in steps),'duplicate_transitions':steps.count(0),
            'received_sps':(len(frames)-valid[0][1])/span if span else None,
            'arrival_span_s':span,'ranges':[[min(r[i] for r in frames),max(r[i] for r in frames)] for i in (1,2,3)]}
        if name=='emg':streams[name]['unexpected_packet_lengths']=unexpected
    with path.open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=streams['emg']['samples']:raise ValueError('Processed/raw EMG sample count mismatch')
    original=cue.get('segments',[]);segments=original
    reviewed=path.with_suffix('.segments_review.json')
    if reviewed.exists():
        segments=json.loads(reviewed.read_text())['segments']
        if any(not (0<=a<b and any(x<=a and b<=y for x,y in original)) for a,b in segments):raise ValueError('Invalid reviewed intervals')
    if any(not 0<=a<b<=len(rows) for a,b in segments):raise ValueError('Invalid segment bounds')
    visuals=[json.loads(Path(p).read_text()) for p in processing.get('visual_cue_files',[]) if p]
    return {'processed_csv':str(path),'processed_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'raw_sha256':hashlib.sha256(Path(raw_files[0]).read_bytes()).hexdigest(),
        'profile':processing['preprocessing_profile'],'streams':streams,
        'processed_values_finite':all(math.isfinite(float(v)) for r in rows for v in r.values()),
        'segments':segments,'selected_samples':sum(b-a for a,b in segments),
        'complete_repetitions':sum(r.get('stop_time') is not None for r in cue['reps']),
        'incomplete_active_repetitions':sum(r.get('stop_time') is None for r in cue['reps']) if cue['mode']=='reps' else 0,
        'visual_status':[v['status'] for v in visuals],
        'user_compliance':'pending','note':'Host arrival rates and cue labels; not calibrated sensor timing or physiological verification.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    result=verify(args.csv);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
