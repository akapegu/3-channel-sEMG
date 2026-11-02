"""Five-second diagnostic of adapted supplier acquisition; not gesture training data."""
import argparse
import asyncio
import csv
import json
import os
import struct
import time
from pathlib import Path
import numpy as np
from bleak import BleakScanner
from record_gesture import DeviceRecorder, merge_and_save, DATA_CHAR_UUID, IMU_CHAR_UUID
from rate_filters import NotchFilter, EXGFilter, PROFILE


async def run(out):
    devices=await BleakScanner.discover(timeout=10,return_adv=True)
    matches=[d for d,a in devices.values() if '27:bc' in (a.local_name or d.name or '').lower()]
    if len(matches)!=1:raise RuntimeError(f'Expected one matching band, found {len(matches)}')
    device=DeviceRecorder(matches[0],None,channels=3,role='dev1')
    print('Connecting to',device.name,flush=True)
    try:
        await device.connect();await device.start_streaming()
        deadline=time.perf_counter()+10
        while not device.rows or not device.has_imu:
            if time.perf_counter()>deadline:raise TimeoutError('Both data streams not received')
            await asyncio.sleep(.02)
        print('Both streams received; recording five seconds. No gestures needed.',flush=True)
        await asyncio.sleep(5)
    finally:
        await device.stop_and_disconnect()
    processed=out/'processed';processed.mkdir()
    result=merge_and_save(str(processed),'pipelinecheck',1,[device],reps=None,mode='continuous')
    logs=[json.loads(line) for line in Path(device.raw_path).read_text().splitlines()]
    datasets={DATA_CHAR_UUID:[],IMU_CHAR_UUID:[]};packet_counts={DATA_CHAR_UUID:0,IMU_CHAR_UUID:0};malformed=0;empty=0
    for packet in logs:
        data=bytes.fromhex(packet['payload_hex']);kind=packet['characteristic']
        if not data:empty+=1;continue
        if len(data)%7:malformed+=1;continue
        packet_id=packet_counts[kind];packet_counts[kind]+=1
        fmt='>BHHH' if kind==DATA_CHAR_UUID else '>Bhhh'
        for within,frame in enumerate(struct.iter_unpack(fmt,data)):
            datasets[kind].append([packet_id,within,packet['host_monotonic_ns'],packet['host_unix_ns'],*frame])
    summary={'device':device.name,'profile':PROFILE,'diagnostic_only':True,'processed_csv':result['csv_path'],
             'raw_packets':device.raw_path,'malformed_packets':malformed,'empty_notifications':empty,'streams':{}}
    for kind,name,columns in [(DATA_CHAR_UUID,'emg',['ch1_adc','ch2_adc','ch3_adc']),(IMU_CHAR_UUID,'accel',['accel_x_counts','accel_y_counts','accel_z_counts'])]:
        rows=datasets[kind]
        with (out/f'{name}_decoded.csv').open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['packet_index','index_in_packet','host_monotonic_ns','host_unix_ns','counter',*columns]);w.writerows(rows)
        first_packet=sum(r[0]==0 for r in rows);span=(rows[-1][2]-rows[0][2])/1e9
        steps=[(b[4]-a[4])%256 for a,b in zip(rows,rows[1:])]
        summary['streams'][name]={'samples':len(rows),'packets':packet_counts[kind],'arrival_span_s':span,
            'received_sps':(len(rows)-first_packet)/span if span else None,'nonunit_counter_transitions':sum(x!=1 for x in steps)}
    nf=[NotchFilter() for _ in range(3)];ef=[EXGFilter() for _ in range(3)]
    for n,e in zip(nf,ef):n.set_sampling_rate(1000);e.set_bits('12',1000)
    replay=np.array([[ef[c].process(nf[c].process(row[5+c],1),4) for c in range(3)] for row in datasets[DATA_CHAR_UUID]])
    actual=np.array([r[2:5] for r in device.rows])
    assert actual.shape==replay.shape
    error=float(np.max(np.abs(actual-replay)))
    summary['max_live_vs_raw_replay_filter_error']=error
    assert error==0
    summary['all_processed_values_finite']=bool(np.isfinite(actual).all())
    summary['note']='Original packets retained; arrival timestamps are not acquisition times. Startup filter transients included. No user gesture labels or accuracy measured.'
    (out/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);os.chdir(out)
    try:asyncio.run(run(out))
    except Exception as exc:
        (out/'failure.json').write_text(json.dumps({'error':str(exc),'time_unix_ns':time.time_ns()},indent=2));raise
