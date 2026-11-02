"""Capture one uninterrupted EMG+LIS3DH stream for all S03 cue blocks."""
import argparse
import asyncio
import json
from pathlib import Path

from bleak import BleakScanner
import record_gesture as recorder
from visual_session import run_guided_session


async def main(args):
    root=Path(recorder.OUTPUT_ROOT)
    subject_dir=root/args.subject
    output_dir=subject_dir/'continuous'
    output_dir.mkdir(parents=True,exist_ok=True)
    capture_trial=len(list(output_dir.glob('guided_session_trial*_merged_*.csv')))+1
    print('Scanning once for the band...',flush=True)
    found=await BleakScanner.discover(timeout=10,return_adv=True)
    matches=[d for d,a in found.values() if '27:bc' in (a.local_name or d.name or '').lower()]
    if len(matches)!=1:raise RuntimeError(f'Expected one 27:BC band; found {len(matches)}')
    devices=recorder.resolve_devices(matches)
    if len(devices)!=1:raise RuntimeError('The continuous adapted session supports one three-channel band only')
    recorders=[]
    for dev,role,channels in devices:
        recorders.append(recorder.DeviceRecorder(dev,out_path=None,channels=channels,role=role))
    print(f'Found {recorders[0].name} ({recorders[0].channels} EMG channels + LIS3DH). '
          'The BLE connection will start after the on-screen Start click.',flush=True)
    cue_path=Path(recorders[0].raw_path).with_suffix('.session_cues.json')
    recorders[0].visual_cue_path=str(cue_path)
    cue_result=None
    session_error=None
    try:
        async def connect_devices():
            for device_recorder in recorders:
                await device_recorder.connect()
        cue_result=await run_guided_session(recorders,cue_path,args.plan,args.seconds_per_task,
                                            connect_callback=connect_devices)
    except Exception as exc:
        session_error=f'{type(exc).__name__}: {exc}'
        print(f'Cue session stopped with error; preserving received samples: {type(exc).__name__}: {exc}',flush=True)
    finally:
        await recorders[0].stop_and_disconnect()
    saved=recorder.merge_and_save(str(output_dir),'guided_session',capture_trial,recorders,reps=[],mode='reps')
    if saved is None:
        print(json.dumps({'subject':args.subject,'continuous_source':None,
            'raw_packet_file':recorders[0].raw_path,'cue_file':str(cue_path),
            'num_samples':0,'cue_status':recorders[0].visual_cue_status,
            'error':session_error or 'No samples were received; no merged CSV was created.'},indent=2),flush=True)
        return
    print(json.dumps({'subject':args.subject,'continuous_source':saved['csv_path'],
        'raw_packet_file':recorders[0].raw_path,'cue_file':str(cue_path),
        'num_samples':saved['num_samples'],'cue_status':recorders[0].visual_cue_status,
        'cue_result':cue_result},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subject',default='abinash_s03')
    p.add_argument('--plan',type=Path,default=Path('training_data/s03_collection_plan.json'))
    p.add_argument('--seconds_per_task',type=int,default=60)
    args=p.parse_args()
    if args.seconds_per_task<30:p.error('--seconds_per_task must be at least 30')
    recorder.VISUAL_CUES=True
    recorder.SAMPLING_RATE=1000
    asyncio.run(main(args))
