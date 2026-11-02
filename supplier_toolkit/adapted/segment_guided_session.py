"""Build class/partition derivatives from one preserved continuous capture.

The source CSV and raw BLE packet log are never modified. All no-command
intervals are gathered into REST for their chronological partition.
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from rate_filters import PROFILE

PARTITIONS = ('train', 'validation', 'test')
CLASSES = ('rest', 'pinch', 'flexion', 'extension')
REACTION_MARGIN_S = 0.5
SAMPLE_RATE = 1000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_intervals(events, n_samples):
    pending = {}
    intervals = []
    for event in events:
        phase_index = event.get('phase_index')
        if phase_index is None or event.get('event') not in ('label_start', 'label_stop'):
            continue
        key = int(phase_index)
        if event['event'] == 'label_start':
            if key in pending:
                raise ValueError(f'Duplicate label start for phase {key}')
            label = event.get('label')
            partition = event.get('partition')
            if label not in CLASSES or partition not in PARTITIONS:
                raise ValueError(f'Invalid class/partition on phase {key}: {label}/{partition}')
            pending[key] = event
            continue
        if key not in pending:
            raise ValueError(f'Label stop without start for phase {key}')
        start_event = pending.pop(key)
        a = int(start_event['sample_index'])
        b = int(event['sample_index'])
        if event.get('label') != start_event.get('label') or event.get('partition') != start_event.get('partition'):
            raise ValueError(f'Label metadata changed within phase {key}')
        if start_event['label'] != 'rest':
            a += round(REACTION_MARGIN_S * SAMPLE_RATE)
        a = max(0, a)
        b = min(n_samples, b)
        if b <= a:
            continue
        intervals.append({
            'partition': start_event['partition'],
            'group': start_event['acquisition_group'],
            'label': start_event['label'],
            'source_start': a,
            'source_stop': b,
            'source_phase_index': key,
            'source_task_id': start_event.get('task_id'),
            'repetition': start_event.get('repetition'),
            'orientation_prompt': start_event.get('orientation_prompt'),
        })
    if pending:
        raise ValueError(f'Unclosed cue-label phases: {sorted(pending)}')
    return intervals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source_csv', type=Path, required=True)
    ap.add_argument('--cue_json', type=Path, required=True)
    ap.add_argument('--subject', default='abinash_s03')
    ap.add_argument('--data_dir', type=Path, default=Path('training_data'))
    ap.add_argument('--output_summary', type=Path, required=True)
    args = ap.parse_args()
    data_root = args.data_dir.resolve()
    source = args.source_csv.resolve()
    cue_path = args.cue_json.resolve()
    cue = json.loads(cue_path.read_text())
    if cue.get('status') != 'complete':
        raise SystemExit('Continuous visual session is incomplete; no labeled derivatives created.')
    processing = json.loads(source.with_suffix('.processing.json').read_text())
    if processing.get('sampling_rate') != SAMPLE_RATE or processing.get('preprocessing_profile') != PROFILE:
        raise SystemExit('Source rate/profile mismatch.')
    source_hash = sha256(source)
    df = pd.read_csv(source)
    intervals = collect_intervals(cue.get('events', []), len(df))
    grouped = {(part, label): [] for part in PARTITIONS for label in CLASSES}
    for interval in intervals:
        grouped[(interval['partition'], interval['label'])].append(interval)
    group_by_partition = {}
    for partition in PARTITIONS:
        groups = {x['group'] for x in intervals if x['partition'] == partition}
        if len(groups) != 1:
            raise ValueError(f'Expected one acquisition group for {partition}, found {sorted(groups)}')
        group_by_partition[partition] = next(iter(groups))
    missing = [f'{part}/{label}' for part in PARTITIONS for label in CLASSES if not grouped[(part, label)]]
    if missing:
        raise ValueError(f'No cue-labeled samples for required class/partition(s): {missing}')
    for values in grouped.values():
        values.sort(key=lambda x: x['source_start'])
        for left, right in zip(values, values[1:]):
            if right['source_start'] < left['source_stop']:
                raise ValueError('Overlapping cue intervals in a derived class file')

    manifest_path = data_root / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'files': [], 'excluded_files': []}
    index_path = data_root / 'dataset_index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {'subjects': {}}
    subject = index.setdefault('subjects', {}).setdefault(args.subject, {})
    source_rel = str(source.relative_to(data_root))
    source_rel_abs = str(source)
    next_trial = {
        label: max((int(x.get('trial', 0)) for x in subject.get(label, {}).get('sessions', [])), default=0) + 1
        for label in CLASSES
    }
    candidates = []

    for partition in PARTITIONS:
        group = group_by_partition[partition]
        for label in CLASSES:
            segments = grouped[(partition, label)]
            chunks = []
            local_segments = []
            source_ranges = []
            reps = []
            cursor = 0
            for interval in segments:
                chunk = df.iloc[interval['source_start']:interval['source_stop']]
                if len(chunk) == 0:
                    continue
                chunks.append(chunk)
                local_segments.append([cursor, cursor + len(chunk)])
                source_ranges.append([interval['source_start'], interval['source_stop']])
                reps.append({
                    'source_task_id': interval['source_task_id'],
                    'source_phase_index': interval['source_phase_index'],
                    'repetition': interval['repetition'],
                    'orientation_prompt': interval['orientation_prompt'],
                    'source_range': [interval['source_start'], interval['source_stop']],
                })
                cursor += len(chunk)
            if not chunks:
                raise ValueError(f'No usable rows for {partition}/{label}')
            local = pd.concat(chunks, ignore_index=True)
            trial = next_trial[label]
            next_trial[label] += 1
            outdir = data_root / args.subject / label
            outdir.mkdir(parents=True, exist_ok=True)
            tag = re.sub(r'^guided_session_trial\d+_', '', source.stem)
            out = outdir / f'{label}_trial{trial}_{tag}_{partition}_derived.csv'
            if out.exists() or out.with_suffix('.beep.json').exists() or out.with_suffix('.processing.json').exists():
                raise FileExistsError(f'Refusing to overwrite derivative for {partition}/{label}: {out}')
            beep = {
                'mode': 'continuous' if label == 'rest' else 'reps',
                'segments': local_segments,
                'segment_labels': [label] * len(local_segments),
                'reps': reps,
                'reaction_delay_sec': REACTION_MARGIN_S if label != 'rest' else 0,
                'total_samples': len(local),
                'label': 'no-command/rest' if label == 'rest' else label,
                'derived_from_source_csv': source_rel_abs,
                'source_sample_ranges': source_ranges,
                'partition': partition,
                'acquisition_group': group,
                'actual_orientations_verified': False,
            }
            meta = {
                'sampling_rate': SAMPLE_RATE,
                'preprocessing_profile': PROFILE,
                'visual_cue_files': [str(cue_path)],
                'visual_cue_status': ['complete'],
                'raw_packet_files': processing.get('raw_packet_files', []),
                'derived_from_source_csv': source_rel_abs,
                'source_csv_sha256': source_hash,
                'source_sample_ranges': source_ranges,
                'derived_class': label,
                'partition': partition,
                'acquisition_group': group,
                'orientation_prompts_only': True,
                'note': 'Rows copied from the preserved continuous source. REST comprises all cued no-target intervals; gesture intervals exclude the first 0.5 s reaction margin. Raw BLE packets remain in the source JSONL.',
            }
            recorded = next((e for e in cue.get('events', []) if e.get('partition') == partition and e.get('gesture') is not None and e.get('event') == 'task_start'), None)
            recorded_at = datetime.fromtimestamp(recorded['host_unix_ns'] / 1e9).astimezone().isoformat(timespec='seconds') if recorded else None
            candidates.append({
                'partition': partition, 'group': group, 'label': label, 'trial': trial,
                'out': out, 'data': local, 'beep': beep, 'meta': meta,
                'segments': local_segments, 'source_ranges': source_ranges,
                'num_samples': len(local), 'valid_samples': sum(b-a for a,b in local_segments),
                'recorded_at': recorded_at, 'reps': reps,
            })

    # Refuse collisions before writing any derivative. The original continuous
    # files remain untouched; these outputs are reproducible derived data.
    for item in candidates:
        item['data'].to_csv(item['out'], index=False, quoting=csv.QUOTE_MINIMAL)
        item['out'].with_suffix('.beep.json').write_text(json.dumps(item['beep'], indent=2) + '\n')
        item['out'].with_suffix('.processing.json').write_text(json.dumps(item['meta'], indent=2) + '\n')
        label_sessions = subject.setdefault(item['label'], {}).setdefault('sessions', [])
        name = item['out'].name
        label_sessions.append({
            'trial': item['trial'], 'csv': name, 'beep_json': item['out'].with_suffix('.beep.json').name,
            'recorded_at': item['recorded_at'], 'mode': 'continuous' if item['label'] == 'rest' else 'reps',
            'num_samples': item['num_samples'], 'num_channels': 3,
            'num_segments': len(item['segments']), 'total_valid_samples': item['valid_samples'],
            'include_in_training': False, 'exclusion_reason': 'Pending user cue-compliance confirmation',
            'partition': item['partition'], 'acquisition_group': item['group'],
            'derived_from_source_csv': source_rel, 'source_sample_ranges': item['source_ranges'],
        })
        relative = str(item['out'].relative_to(data_root))
        manifest['files'].append({'path': relative, 'partition': item['partition'], 'acquisition_group': item['group']})

    index_path.write_text(json.dumps(index, indent=2) + '\n')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    created = [{
        'label': x['label'], 'partition': x['partition'], 'acquisition_group': x['group'],
        'trial': x['trial'], 'csv': str(x['out']), 'source_sample_ranges': x['source_ranges'],
        'selected_segments': x['segments'], 'selected_samples': x['valid_samples'],
        'orientation_prompts_only': True,
    } for x in candidates]
    summary = {
        'status': 'segmented_pending_user_compliance', 'source_csv': str(source),
        'source_csv_sha256': source_hash, 'cue_json': str(cue_path),
        'source_rows': len(df), 'source_raw_packet_files': processing.get('raw_packet_files', []),
        'raw_source_untouched': True, 'derived_files': created,
        'note': 'One continuous source split into cue-labeled class/partition derivatives. Every non-target cue interval maps to REST; test is chronologically separate within the same wearing/session.',
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'derived_count': len(created), 'summary': str(args.output_summary), 'files': created}, indent=2))


if __name__ == '__main__':
    main()
