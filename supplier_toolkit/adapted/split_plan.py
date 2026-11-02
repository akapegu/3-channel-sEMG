"""Explicit whole-acquisition split required before creating overlapping windows."""
import json
from pathlib import Path
PARTITIONS=('train','validation','test')


def load_plan(path,root):
    root=Path(root).resolve();data=json.loads(Path(path).read_text());files={};groups={}
    for entry in data['files']:
        name=(root/entry['path']).resolve()
        if not name.is_relative_to(root):raise ValueError('Split path outside dataset')
        part=entry['partition'];group=entry['acquisition_group']
        if part not in PARTITIONS:raise ValueError('Unknown partition')
        if name in files:raise ValueError('Duplicate file in split plan')
        if group in groups and groups[group]!=part:raise ValueError('Acquisition group crosses partitions')
        groups[group]=part;files[name]=part
    return files


def partition_for(plan,path):
    return plan[Path(path).resolve()]


def verify_processing(path,profile):
    metadata=Path(path).with_suffix('.processing.json')
    data=json.loads(metadata.read_text())
    if any(status not in (None,'complete') for status in data.get('visual_cue_status',[])):
        raise ValueError(f'Incomplete visual trial requires exclusion/review: {path}')
    if data.get('sampling_rate')!=1000 or data.get('preprocessing_profile')!=profile:
        raise ValueError(f'Processing mismatch: {path}')
