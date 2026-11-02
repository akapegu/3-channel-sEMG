"""Read-only per-class diagnostics for S03 TRAIN and VALIDATION only.

This script deliberately excludes every TEST path before opening CSV files.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow import keras

PROJECT = Path(__file__).resolve().parents[2]
ADAPTED = PROJECT / 'supplier_toolkit' / 'adapted'
sys.path.insert(0, str(ADAPTED))
from train_gesture_model import get_channel_columns, load_gesture_files, window_dataframe
from split_plan import load_plan, partition_for


def main():
    data_dir = ADAPTED / 'training_data'
    manifest_path = data_dir / 'split_manifest.json'
    model_dir = Path(__file__).resolve().parent / 's03_cnn_lstm_weighted_v1'
    meta = json.loads((model_dir / 'meta.json').read_text())
    plan = load_plan(manifest_path, data_dir)
    dev_plan = {path: part for path, part in plan.items() if part in ('train', 'validation')}
    allowed = {os.path.realpath(path) for path in dev_plan}
    items, _, _ = load_gesture_files(str(data_dir), 'abinash_s03', allowed_paths=allowed)
    model = keras.models.load_model(model_dir / 'model.keras')
    classes = meta['classes']
    mean = np.asarray(meta['scaler_mean'], dtype=np.float32)
    std = np.asarray(meta['scaler_std'], dtype=np.float32)
    results = {}
    for part in ('train', 'validation'):
        y_true, y_pred = [], []
        per_class_windows = {}
        for label, frame, csv_path, segments in items:
            if partition_for(dev_plan, csv_path) != part:
                continue
            windows = window_dataframe(frame, get_channel_columns(frame),
                                       meta['window_size'], meta['stride'], segments)
            if not len(windows):
                continue
            normalized = (windows - mean) / std
            pred = np.argmax(model.predict(normalized, verbose=0), axis=1)
            true = np.full(len(windows), classes.index(label), dtype=np.int64)
            y_true.extend(true.tolist())
            y_pred.extend(pred.tolist())
            per_class_windows[label] = per_class_windows.get(label, 0) + len(windows)
        report = classification_report(y_true, y_pred, labels=range(len(classes)),
                                       target_names=classes, output_dict=True,
                                       zero_division=0)
        results[part] = {
            'window_count': len(y_true),
            'class_window_counts': per_class_windows,
            'accuracy': report['accuracy'],
            'macro_f1': f1_score(y_true, y_pred, labels=range(len(classes)),
                                 average='macro', zero_division=0),
            'classification_report': report,
            'confusion_matrix_true_rows_predicted_columns': confusion_matrix(
                y_true, y_pred, labels=range(len(classes))).tolist(),
        }
    output = {
        'model_dir': str(model_dir),
        'classes': classes,
        'window_size': meta['window_size'],
        'stride': meta['stride'],
        'partitions_opened': ['train', 'validation'],
        'test_files_opened': False,
        'results': results,
    }
    out_path = Path(__file__).resolve().parent / 's03_train_validation_diagnostics.json'
    out_path.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({
        part: {'window_count': v['window_count'], 'accuracy': v['accuracy'],
               'macro_f1': v['macro_f1'], 'per_class': {
                   k: {'precision': round(v['classification_report'][k]['precision'], 3),
                       'recall': round(v['classification_report'][k]['recall'], 3),
                       'f1': round(v['classification_report'][k]['f1-score'], 3)}
                   for k in classes}}
        for part, v in results.items()
    }, indent=2))
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
