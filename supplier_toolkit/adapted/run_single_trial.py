"""One supplier recording with visible Start/Stop controls; no terminal prompts."""
import argparse
import asyncio
import json
from pathlib import Path
from bleak import BleakScanner
import record_gesture as recorder


async def main(args):
    root=Path(recorder.OUTPUT_ROOT)
    print('Scanning for the band...',flush=True)
    found=await BleakScanner.discover(timeout=10,return_adv=True)
    matches=[d for d,a in found.values() if '27:bc' in (a.local_name or d.name or '').lower()]
    if len(matches)!=1:raise RuntimeError(f'Expected one 27:BC band; found {len(matches)}')
    devices=recorder.resolve_devices(matches)
    subject_dir=root/args.subject;subject_dir.mkdir(parents=True,exist_ok=True)
    trial=recorder.next_trial_number(str(subject_dir),args.gesture)
    print(f'{args.subject} / {args.gesture} / trial {trial} / {args.partition} / {args.group}',flush=True)
    result=await recorder.record_gesture(args.gesture,devices,trial,str(subject_dir))
    index=recorder.load_dataset_index(str(root))
    recorder.register_session(index,args.subject,args.gesture,result,trial)
    entry=next(x for x in index['subjects'][args.subject][args.gesture]['sessions'] if x['trial']==trial)
    entry.update(include_in_training=False,exclusion_reason='Pending user cue compliance confirmation',partition=args.partition,acquisition_group=args.group)
    recorder.save_dataset_index(str(root),index)
    path=Path(result['csv_path'])
    manifest_path=root/'split_manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files':[]}
    relative=str(path.relative_to(root))
    if any(x['path']==relative for x in manifest['files']):raise ValueError('File already assigned')
    manifest['files'].append({'path':relative,'partition':args.partition,'acquisition_group':args.group})
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'file':str(path),'trial':trial,'partition':args.partition,'pending_user_confirmation':True,'result':result},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',required=True);p.add_argument('--gesture',required=True,choices=('rest','pinch','flexion','extension'))
    p.add_argument('--partition',required=True,choices=('train','validation','test'));p.add_argument('--group',required=True)
    p.add_argument('--duration',type=int,default=30,help='hard recording cap in seconds')
    p.add_argument('--orientation_sequence',default='neutral',
                   help='comma-separated forearm positions: neutral,palm_up,palm_down')
    args=p.parse_args()
    if not args.subject or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.subject):p.error('Use a simple subject folder name')
    if args.duration < 20:p.error('Duration must be at least 20 seconds')
    allowed={'neutral','palm_up','palm_down'}
    args.orientation_sequence=[x.strip() for x in args.orientation_sequence.split(',') if x.strip()]
    if not args.orientation_sequence or any(x not in allowed for x in args.orientation_sequence):
        p.error('Use a comma-separated sequence of neutral,palm_up,palm_down')
    recorder.VISUAL_CUES=True;recorder.MAX_RECORD_SECONDS=30;recorder.REACTION_DELAY_SEC=.5
    recorder.GESTURE_HOLD_MIN_SEC=3;recorder.GESTURE_HOLD_MAX_SEC=5
    recorder.REP_INTERVAL_SEC=6;recorder.REP_INTERVAL_JITTER_SEC=.5
    recorder.MAX_RECORD_SECONDS=args.duration
    recorder.VISUAL_ORIENTATION_SEQUENCE=args.orientation_sequence
    asyncio.run(main(args))
