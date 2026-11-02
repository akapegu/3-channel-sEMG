"""Score a locked model on one fresh cue-labeled test-only acquisition.

This script performs no fitting, normalization updates, or threshold tuning.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras

from rate_filters import PROFILE

EMG_COLUMNS = ('ch1', 'ch2', 'ch3')
IMU_COLUMNS = ('accel_x', 'accel_y', 'accel_z')
PARTITIONS = ('train', 'validation', 'test')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cue_intervals(events, n_samples, expected_acquisition):
    pending = {}
    intervals = []
    for event in events:
        if event.get('event') not in ('label_start', 'label_stop'):
            continue
        phase = event.get('phase_index')
        if phase is None:
            raise ValueError('Labeled event lacks phase_index')
        phase = int(phase)
        if event['event'] == 'label_start':
            if phase in pending:
                raise ValueError(f'Duplicate label start for phase {phase}')
            if event.get('partition') != 'test':
                raise ValueError('Locked-model evaluator accepts test-only cue phases')
            if event.get('acquisition_group') != expected_acquisition:
                raise ValueError('Cue acquisition group does not match the requested fresh test')
            pending[phase] = event
            continue
        start = pending.pop(phase, None)
        if start is None:
            raise ValueError(f'Label stop without start for phase {phase}')
        if event.get('label') != start.get('label'):
            raise ValueError(f'Label changed within phase {phase}')
        a, b = int(start['sample_index']), int(event['sample_index'])
        if start['label'] != 'rest':
            a += 500  # fixed 0.5 s reaction margin at 1000 SPS
        a, b = max(0, a), min(n_samples, b)
        if b > a:
            intervals.append({'label': start['label'], 'start': a, 'stop': b,
                              'phase_index': phase})
    if pending:
        raise ValueError(f'Unclosed labels: {sorted(pending)}')
    return intervals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source_csv', type=Path, required=True)
    ap.add_argument('--cue_json', type=Path, required=True)
    ap.add_argument('--model_dir', type=Path, required=True)
    ap.add_argument('--acquisition_group', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    source = args.source_csv.resolve()
    model_dir = args.model_dir.resolve()
    cue = json.loads(args.cue_json.resolve().read_text())
    processing = json.loads(source.with_suffix('.processing.json').read_text())
    meta = json.loads((model_dir / 'meta.json').read_text())
    if cue.get('status') != 'complete' or cue.get('evaluation_only') is not True:
        raise ValueError('Cue log must be complete and explicitly evaluation-only')
    if cue.get('partitions') != ['test']:
        raise ValueError('Cue log contains non-test partitions')
    if processing.get('sampling_rate') != meta.get('sampling_rate'):
        raise ValueError('Sampling rate differs from locked model')
    if processing.get('preprocessing_profile') != meta.get('preprocessing_profile') or PROFILE != meta.get('preprocessing_profile'):
        raise ValueError('Preprocessing profile differs from locked model')
    if any(status not in (None, 'complete') for status in processing.get('visual_cue_status', [])):
        raise ValueError('Incomplete visual capture cannot be evaluated')

    frame = pd.read_csv(source)
    columns = meta['channel_columns']
    if columns != list(EMG_COLUMNS + IMU_COLUMNS):
        raise ValueError(f'Unexpected feature columns in locked model: {columns}')
    if not set(columns).issubset(frame.columns):
        raise ValueError('Source CSV is missing one or more model input columns')
    intervals = cue_intervals(cue.get('events', []), len(frame), args.acquisition_group)
    labels = set(meta['classes'])
    if {x['label'] for x in intervals} != labels:
        raise ValueError(f'Test cue classes do not match model classes: {labels} vs {sorted({x["label"] for x in intervals})}')

    window_size = int(meta['window_size'])
    stride = int(meta['stride'])
    mean = np.asarray(meta['scaler_mean'], dtype=np.float32)
    std = np.asarray(meta['scaler_std'], dtype=np.float32)
    std[std == 0] = 1.0
    class_to_id = {name: i for i, name in enumerate(meta['classes'])}
    all_windows, y_true = [], []
    interval_window_ranges = []
    for interval in intervals:
        data = frame.iloc[interval['start']:interval['stop']][columns].to_numpy(dtype=np.float32)
        begin = len(all_windows)
        for start in range(0, max(0, len(data) - window_size + 1), stride):
            all_windows.append(data[start:start + window_size])
            y_true.append(class_to_id[interval['label']])
        interval_window_ranges.append({'label': interval['label'], 'start': interval['start'],
                                       'stop': interval['stop'], 'window_start': begin,
                                       'window_stop': len(all_windows)})
    if not all_windows:
        raise ValueError('No complete test windows fit inside cue intervals')
    X = (np.stack(all_windows) - mean) / std
    model = keras.models.load_model(model_dir / 'model.keras')
    y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
    y_true = np.asarray(y_true, dtype=np.int64)
    class_ids = range(len(meta['classes']))
    report = classification_report(y_true, y_pred, labels=class_ids,
                                   target_names=meta['classes'], output_dict=True,
                                   zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)

    rest_id = class_to_id['rest']
    rest_windows = false_windows = false_bouts = 0
    rest_seconds = 0.0
    rest_sequences = []
    for entry in interval_window_ranges:
        if entry['label'] != 'rest':
            continue
        a, b = entry['window_start'], entry['window_stop']
        predictions = y_pred[a:b].tolist()
        rest_windows += len(predictions)
        false_windows += sum(p != rest_id for p in predictions)
        rest_seconds += max(0, entry['stop'] - entry['start']) / meta['sampling_rate']
        if predictions:
            rest_sequences.append((entry['start'], entry['stop'], predictions))
    rest_sequences.sort(key=lambda x: x[0])
    merged_sequences = []
    for start, stop, predictions in rest_sequences:
        if merged_sequences and start <= merged_sequences[-1][1]:
            merged_sequences[-1][1] = max(merged_sequences[-1][1], stop)
            merged_sequences[-1][2].extend(predictions)
        else:
            merged_sequences.append([start, stop, list(predictions)])
    for _, _, predictions in merged_sequences:
        active = False
        for prediction in predictions:
            now_active = prediction != rest_id
            if now_active and not active:
                false_bouts += 1
            active = now_active

    result = {
        'evaluation_only': True,
        'model_dir': str(model_dir),
        'model_sha256': sha256(model_dir / 'model.keras'),
        'source_csv': str(source),
        'source_csv_sha256': sha256(source),
        'cue_json': str(args.cue_json.resolve()),
        'acquisition_group': args.acquisition_group,
        'sampling_rate': meta['sampling_rate'],
        'preprocessing_profile': meta['preprocessing_profile'],
        'window_size': window_size,
        'stride': stride,
        'feature_columns': columns,
        'test_windows': len(y_true),
        'accuracy': float(report['accuracy']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'classification_report': report,
        'confusion_matrix_true_rows_predicted_columns': matrix.tolist(),
        'rest_window_count': rest_windows,
        'rest_false_active_window_count': false_windows,
        'rest_false_active_window_fraction': false_windows / rest_windows if rest_windows else None,
        'rest_false_active_bouts': false_bouts,
        'evaluated_rest_seconds': rest_seconds,
        'rest_false_active_bouts_per_minute': false_bouts / (rest_seconds / 60) if rest_seconds else None,
        'note': 'Locked model scored once without fitting or threshold changes. Overlapping windows and window-level false actives are correlated; false-active bouts/min is an offline ungated estimate, not a measured device command rate. Orientation prompts are not verified actual posture.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('test_windows','accuracy','macro_f1',
        'rest_window_count','rest_false_active_window_count','rest_false_active_window_fraction',
        'rest_false_active_bouts','rest_false_active_bouts_per_minute')}, indent=2))


if __name__ == '__main__':
    main()
