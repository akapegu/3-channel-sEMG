"""Synthetic verification; no BLE access or device recording."""
import csv
import json
import struct
import tempfile
from pathlib import Path
from npg_accelerometer import AccelerometerCapture, decode_accelerometer

assert decode_accelerometer(struct.pack('>Bhhh',255,-32768,0,32767)) == [(255,-32768,0,32767)]
assert decode_accelerometer(b'') == []
capture=AccelerometerCapture()
capture.notify(None, struct.pack('>Bhhh',255,-32768,0,32767)+struct.pack('>Bhhh',0,-1,2,-3))
capture.notify(None, struct.pack('>Bhhh',1,4,5,6))
assert len(capture.rows)==3 and capture.gaps==0 and capture.duplicates==0
assert capture.rows[0][3:5]==capture.rows[1][3:5]
capture.notify(None,b'')
capture.notify(None,b'bad')
assert capture.empty==1 and capture.malformed==1
capture.notify(None,struct.pack('>Bhhh',1,0,0,0))
capture.notify(None,struct.pack('>Bhhh',4,0,0,0))
assert capture.duplicates==1 and capture.gaps==2
assert capture.first.is_set()
with tempfile.TemporaryDirectory() as folder:
    output=Path(folder)/'synthetic.csv'
    capture.save(output,'SYNTHETIC')
    with output.with_suffix('.imu.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==5 and int(rows[0]['accel_x_counts'])==-32768
    stats=json.loads(output.with_suffix('.imu.json').read_text())
    assert stats['samples_per_axis']==5 and stats['counter_gap_missing_modulo_256']==2
    try:
        capture.save(output,'SYNTHETIC')
    except FileExistsError:
        pass
    else:
        raise AssertionError('Existing output overwritten')
print('Synthetic accelerometer checks passed: signed decoding, batches, rollover, gaps, duplicates, empty/malformed payloads, CSV/JSON preservation, overwrite refusal.')
