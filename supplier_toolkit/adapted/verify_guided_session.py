"""Read-only packet/rate QC for a one-band continuous guided capture."""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

EMG_UUID = 'beb5483e-36e1-4688-b7f5-ea07361b26a8'
IMU_UUID = '5a153fa9-7be0-400c-8ef8-d84502b31c4d'


def inspect_stream(packets, characteristic, fmt, expected_packet_bytes=None):
    selected = [p for p in packets if p.get('characteristic', '').lower() == characteristic]
    frames, arrivals = [], []
    empty = malformed = unexpected = 0
    for packet in selected:
        payload = bytes.fromhex(packet['payload_hex'])
        if not payload:
            empty += 1
            continue
        if len(payload) % 7:
            malformed += 1
            continue
        if expected_packet_bytes is not None and len(payload) != expected_packet_bytes:
            unexpected += 1
        decoded = list(struct.iter_unpack(fmt, payload))
        frames.extend(decoded)
        arrivals.extend([int(packet['host_monotonic_ns'])] * len(decoded))
    if not frames:
        raise ValueError(f'No valid frames for characteristic {characteristic}')
    steps = [(b[0] - a[0]) % 256 for a, b in zip(frames, frames[1:])]
    span_s = (arrivals[-1] - arrivals[0]) / 1e9
    return {
        'notification_count': len(selected),
        'frame_count': len(frames),
        'empty_notifications': empty,
        'malformed_notifications': malformed,
        'unexpected_packet_lengths': unexpected,
        'counter_nonunit_transitions': sum(step != 1 for step in steps),
        'counter_duplicate_transitions': steps.count(0),
        'counter_gap_transitions': sum(step not in (0, 1) for step in steps),
        'counter_first_last': [frames[0][0], frames[-1][0]],
        'arrival_span_seconds': span_s,
        'host_arrival_rate_hz': (len(frames) - 1) / span_s if span_s > 0 else None,
        'value_ranges': [[min(row[i] for row in frames), max(row[i] for row in frames)] for i in (1, 2, 3)],
    }


def verify(source_csv):
    source_csv = Path(source_csv).resolve()
    processing_path = source_csv.with_suffix('.processing.json')
    processing = json.loads(processing_path.read_text())
    raw_paths = processing.get('raw_packet_files', [])
    if len(raw_paths) != 1:
        raise ValueError(f'Expected one raw packet file, found {len(raw_paths)}')
    raw_path = Path(raw_paths[0]).resolve()
    packets = [json.loads(line) for line in raw_path.read_text().splitlines() if line.strip()]
    streams = {
        'emg': inspect_stream(packets, EMG_UUID, '>BHHH', expected_packet_bytes=140),
        'accelerometer': inspect_stream(packets, IMU_UUID, '>Bhhh'),
    }
    with source_csv.open(newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)
    if not {'counter_dev1', 'ch1', 'ch2', 'ch3', 'accel_x', 'accel_y', 'accel_z'}.issubset(columns):
        raise ValueError(f'Unexpected source columns: {columns}')
    if len(rows) != streams['emg']['frame_count']:
        raise ValueError(f"Processed rows ({len(rows)}) != decoded raw EMG samples ({streams['emg']['frame_count']})")
    finite = all(math.isfinite(float(value)) for row in rows for value in row.values())
    if not finite:
        raise ValueError('Non-finite value in processed source')
    cue_path = Path(processing.get('visual_cue_files', [None])[0]).resolve()
    cue = json.loads(cue_path.read_text()) if cue_path and cue_path.exists() else None
    cue_qc = None
    if cue:
        tasks = {}
        for event in cue.get('events', []):
            if event.get('task_id') is not None:
                tasks.setdefault(event['task_id'], []).append(event)
        indices = [int(e['sample_index']) for e in cue.get('events', [])]
        cue_qc = {
            'status': cue.get('status'),
            'event_count': len(cue.get('events', [])),
            'task_count': len(tasks),
            'all_event_sample_indices_in_bounds': all(0 <= i <= len(rows) for i in indices),
            'task_ids': sorted(tasks),
        }
    return {
        'source_csv': str(source_csv),
        'source_csv_sha256': hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        'raw_packet_file': str(raw_path),
        'raw_packet_sha256': hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        'sampling_rate_configured_hz': processing.get('sampling_rate'),
        'preprocessing_profile': processing.get('preprocessing_profile'),
        'processed_sample_count': len(rows),
        'processed_values_finite': finite,
        'streams': streams,
        'cue_qc': cue_qc,
        'note': 'Host notification-arrival rate and sample-counter continuity; host timing is not calibrated sensor-clock timing.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
