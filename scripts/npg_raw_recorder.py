#!/usr/bin/env python3
"""NPG 3-channel BLE recorder. pip install bleak; python npg_raw_recorder.py
No digital filtering or voltage conversion. Firmware may already scale ADC counts.
Arrival times are host notification times, NOT individual acquisition timestamps.
"""
import argparse
import asyncio
import csv
import json
import struct
import time
from pathlib import Path

DATA = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'
CONTROL = '0000ff01-0000-1000-8000-00805f9b34fb'


def decode_packet(data):
    if not data or len(data) % 7:
        raise ValueError(f'Expected 7-byte sample frames; received {len(data)} bytes')
    return list(struct.iter_unpack('>BHHH', data))


async def record(args):
    from bleak import BleakClient, BleakScanner
    print('Scanning... Close/disconnect the browser visualizer first.')
    found = await BleakScanner.discover(timeout=10, return_adv=True)
    candidates = [(d, a.local_name or d.name or '') for d, a in found.values()
                  if 'npg' in (a.local_name or d.name or '').lower()]
    if args.name:
        candidates = [(d, n) for d, n in candidates if args.name.lower() in n.lower()]
    if not candidates:
        raise RuntimeError('No matching NPG found. Check power, Bluetooth permission, and other connected apps.')
    for i, (_, n) in enumerate(candidates, 1):
        print(f'{i}: {n}')
    pick = 0 if len(candidates) == 1 else int(input('Device number: ')) - 1
    if not 0 <= pick < len(candidates):
        raise ValueError('Invalid device number')
    device, name = candidates[pick]
    rows, packets = [], []
    state = {'previous': None, 'gaps': 0, 'duplicates': 0, 'malformed': 0}
    first = asyncio.Event()
    motion = None
    if args.imu:
        from npg_accelerometer import AccelerometerCapture, IMU
        motion = AccelerometerCapture()

    def notify(_, payload):
        mono, unix = time.perf_counter_ns(), time.time_ns()
        try:
            frames = decode_packet(payload)
        except ValueError as exc:
            state['malformed'] += 1
            print(f'WARNING: {exc}')
            return
        packet_id = len(packets)
        packets.append((mono, len(frames)))
        for within, (counter, ch1, ch2, ch3) in enumerate(frames):
            previous = state['previous']
            step = None if previous is None else (counter - previous) % 256
            if step == 0:
                state['duplicates'] += 1
            elif step is not None and step != 1:
                state['gaps'] += step - 1
            state['previous'] = counter
            rows.append((len(rows), packet_id, within, mono, unix, counter, ch1, ch2, ch3))
        first.set()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() or args.out.with_suffix('.json').exists():
        raise FileExistsError('Output exists; choose another --out filename.')
    if args.cued and args.out.with_suffix('.cues.json').exists():
        raise FileExistsError('Cue output exists; choose another --out filename.')
    if motion and any(args.out.with_suffix(s).exists() for s in ('.imu.csv', '.imu.json')):
        raise FileExistsError('Accelerometer output exists; choose another --out filename.')
    try:
        async with BleakClient(device, timeout=20) as client:
            await client.start_notify(DATA, notify)
            motion_subscribed = False
            try:
                if motion:
                    await client.start_notify(IMU, motion.notify)
                    motion_subscribed = True
                await client.write_gatt_char(CONTROL, b'START', response=True)
                await asyncio.wait_for(first.wait(), timeout=10)
                if motion:
                    await asyncio.wait_for(motion.first.wait(), timeout=10)
                    print('EMG and accelerometer data received.')
                if args.cued:
                    from emg_cues import run_cues
                    action={'tilt':'GENTLE FOREARM TILT','clench':'GENTLE CLENCH','mixed':'MIXED'}[args.cue_action]
                    await run_cues(client, rows, args.out.with_suffix('.cues.json'), args.clench_seconds, args.relax_seconds, action)
                    return
                print(f'Recording {args.seconds:g} seconds. Relax initially, then perform your movements.')
                end = time.perf_counter() + args.seconds
                while time.perf_counter() < end:
                    await asyncio.sleep(min(1, max(0.01, end - time.perf_counter())))
                    if not client.is_connected:
                        print('Device disconnected; saving partial recording.')
                        break
            finally:
                if client.is_connected:
                    try:
                        await client.write_gatt_char(CONTROL, b'STOP', response=True)
                    finally:
                        try:
                            await client.stop_notify(DATA)
                        finally:
                            if motion_subscribed:
                                await client.stop_notify(IMU)
    finally:
        if rows:
            with args.out.open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['sample_index', 'packet_index', 'index_in_packet',
                                 'host_monotonic_ns', 'host_unix_ns', 'counter',
                                 'ch1_adc', 'ch2_adc', 'ch3_adc'])
                writer.writerows(rows)
            elapsed = (packets[-1][0] - packets[0][0]) / 1e9
            # Exclude first packet because its acquisition precedes its arrival.
            rate = (len(rows) - packets[0][1]) / elapsed if elapsed > 0 else None
            stats = {'device': name, 'samples_per_channel': len(rows),
                     'packets': len(packets), 'host_arrival_span_seconds': elapsed,
                     'estimated_received_sps_per_channel': rate,
                     'counter_gap_missing_modulo_256': state['gaps'],
                     'duplicate_counter_transitions': state['duplicates'],
                     'malformed_packets': state['malformed'],
                     'channel_ranges': [[min(r[i] for r in rows), max(r[i] for r in rows)] for i in (6,7,8)],
                     'note': 'No host filters. Firmware scaling may apply. Host arrival times are not acquisition timestamps. 8-bit counters cannot detect losses of whole 256-frame cycles; gaps also need interpretation for resets/reordering. Rate is received throughput, not a calibrated ADC clock measurement.'}
            args.out.with_suffix('.json').write_text(json.dumps(stats, indent=2))
            print(json.dumps(stats, indent=2))
            print(f'Saved {args.out} and {args.out.with_suffix(".json")}')
        if motion:
            motion.save(args.out, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seconds', type=float, default=35)
    ap.add_argument('--out', type=Path, default=Path('npg_raw_' + time.strftime('%Y%m%d_%H%M%S') + '.csv'))
    ap.add_argument('--name', help='Optional device-name substring when multiple bands are nearby')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--imu', action='store_true', help='Also preserve every accelerometer frame in separate .imu.csv/.imu.json files; require live motion notifications')
    ap.add_argument('--cued', action='store_true', help='Visible rest/clench protocol after 5-second preparation; overrides --seconds')
    ap.add_argument('--clench-seconds', type=float, default=2, help='Each cued clench duration (default 2)')
    ap.add_argument('--relax-seconds', type=float, default=3, help='Relax duration following each clench (default 3)')
    ap.add_argument('--cue-action', choices=('clench','tilt','mixed'), default='clench', help='Cued action; mixed interleaves four clenches/four loose-hand tilts')
    args = ap.parse_args()
    if args.self_test:
        assert decode_packet(struct.pack('>BHHH', 255, 0, 2048, 4095)) == [(255,0,2048,4095)]
        assert decode_packet(struct.pack('>BHHH', 0, 123, 456, 789)*20) == [(0,123,456,789)]*20
        try:
            decode_packet(b'bad')
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed payload accepted')
        print('Packet decoding checks passed.')
        return
    if not all(0 < value < float('inf') for value in (args.seconds, args.clench_seconds, args.relax_seconds)):
        ap.error('Durations must be finite and positive')
    try:
        asyncio.run(record(args))
    except KeyboardInterrupt:
        print('Stopped.')
    except Exception as exc:
        raise SystemExit(f'ERROR: {exc}')

if __name__ == '__main__':
    main()
