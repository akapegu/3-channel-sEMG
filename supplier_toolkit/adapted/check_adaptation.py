"""Synthetic checks only. No Bluetooth access or user-data training."""
import json
import os
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.signal import butter,sosfilt,sosfreqz
from rate_filters import CausalSOS,NotchFilter,EXGFilter,PROFILE
from split_plan import load_plan

rng=np.random.default_rng(17);values=rng.normal(size=4000)
for sos in [butter(2,70,btype='highpass',fs=1000,output='sos'),butter(2,[48,52],btype='bandstop',fs=1000,output='sos')]:
 state=CausalSOS(sos);observed=np.array([state.process(v) for v in values]);assert np.allclose(observed,sosfilt(sos,values),atol=1e-11)
notch=NotchFilter();notch.set_sampling_rate(1000);hp=EXGFilter();hp.set_bits('12',1000)
combined=np.array([hp.process(notch.process(v,1),4) for v in values])
assert np.allclose(combined,sosfilt(butter(2,70,btype='highpass',fs=1000,output='sos'),sosfilt(butter(2,[48,52],btype='bandstop',fs=1000,output='sos'),values)),atol=1e-10)
_,response=sosfreqz(butter(2,[48,52],btype='bandstop',fs=1000,output='sos'),worN=[50,100],fs=1000)
assert abs(response[0])<.001 and abs(response[1])>.99
# Importing modules must not connect to a band or begin model training.
import record_gesture as recorder
import gesture_ui_server as dashboard
assert recorder.NotchFilter is dashboard.NotchFilter
assert recorder.EXGFilter is dashboard.EXGFilter
assert recorder.SAMPLING_RATE==1000
with tempfile.TemporaryDirectory() as folder:
 previous=os.getcwd();os.chdir(folder)
 try:
  device=recorder.DeviceRecorder(SimpleNamespace(name='SYNTHETIC',address='synthetic'),None,channels=3,role='test')
  emg=struct.pack('>BHHH',0,1000,2000,3000)*20
  imu=struct.pack('>Bhhh',255,-32768,2,32767)+struct.pack('>Bhhh',0,-3,4,5)
  device.handle_notify(None,emg);device.handle_imu_notify(None,imu);device.raw_file.close()
  packets=[json.loads(x) for x in Path(device.raw_path).read_text().splitlines()]
  assert bytes.fromhex(packets[0]['payload_hex'])==emg
  assert bytes.fromhex(packets[1]['payload_hex'])==imu
  assert len(device.rows)==20 and device.latest_accel==[-3,4,5]
  merged=recorder.merge_and_save(folder,'rest',1,[device],reps=None,mode='continuous')
  meta=json.loads(Path(merged['csv_path']).with_suffix('.processing.json').read_text())
  assert meta['sampling_rate']==1000 and meta['preprocessing_profile']==PROFILE
  manifest=Path(folder)/'split.json';manifest.write_text(json.dumps({'files':[{'path':'a.csv','partition':'train','acquisition_group':'g'},{'path':'b.csv','partition':'test','acquisition_group':'g'}]}))
  try:load_plan(manifest,folder)
  except ValueError:pass
  else:raise AssertionError('Acquisition group leakage accepted')
 finally:os.chdir(previous)
print('PASS: causal filter/scipy equivalence, correct notch frequency, shared record/live filters, native rate, byte-exact raw EMG/IMU logging, metadata, group leakage rejection.')
