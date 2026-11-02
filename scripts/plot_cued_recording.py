"""Plot EXP-004 received samples against host cue times; no acquisition-time claim."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input', type=Path, default=Path('recordings/exp004_attempt02_raw.csv'))
args=parser.parse_args()
p=args.input
with p.open() as f:
    rows=list(csv.DictReader(f))
cues=json.loads(p.with_suffix('.cues.json').read_text())
events=cues['events']
start=next(e['host_monotonic_ns'] for e in events if e['action']=='RELAX')
end=(next(e['host_monotonic_ns'] for e in events if e['action']=='DONE')-start)/1e9
t=np.array([(int(r['host_monotonic_ns'])-start)/1e9 for r in rows])
x=np.array([[int(r[f'ch{c}_adc']) for c in (1,2,3)] for r in rows])
labels=np.full(len(rows),'outside_protocol',dtype=object)
imu_path=p.with_suffix('.imu.csv')
has_imu=imu_path.exists()
fig,axes=plt.subplots(4 if has_imu else 3,1,figsize=(14,12 if has_imu else 9),sharex=True,layout='constrained')
for i,e in enumerate(events):
    a=(e['host_monotonic_ns']-start)/1e9
    b=(events[i+1]['host_monotonic_ns']-start)/1e9 if i+1<len(events) else a
    if e['action'] in ('RELAX','GENTLE CLENCH','GENTLE FOREARM TILT'):
        labels[(t>=a)&(t<b)]=e['action']
    if e['action'] in ('GENTLE CLENCH','GENTLE FOREARM TILT'):
        color='#e6a23c' if e['action']=='GENTLE CLENCH' else '#77b8ac'
        for ax in axes:
            ax.axvspan(a,b,color=color,alpha=.24)
        prefix='C' if e['action']=='GENTLE CLENCH' else 'T'
        axes[0].text((a+b)/2,.94,prefix+str(e['repetition']),transform=axes[0].get_xaxis_transform(),ha='center')
for c,ax in enumerate(axes[:3]):
    ax.plot(t,x[:,c],lw=.45,color='#2864a0')
    ax.set_ylabel(f'CH{c+1} received counts')
    ax.grid(alpha=.2)
    ax.set_xlim(-1,end+1)
    visible=x[(t>=-1)&(t<=end+1),c]
    lo,hi=visible.min(),visible.max()
    pad=max(10,(hi-lo)*.08)
    ax.set_ylim(lo-pad,hi+pad)
if has_imu:
    with imu_path.open() as f:
        motion=list(csv.DictReader(f))
    mt=np.array([(int(r['host_monotonic_ns'])-start)/1e9 for r in motion])
    visible_motion=[]
    for axis in ('x','y','z'):
        v=np.array([int(r[f'accel_{axis}_counts']) for r in motion])
        keep=(mt>=0)&(mt<10)
        offset=np.median(v[keep])
        axes[3].plot(mt,v-offset,label=f'{axis.upper()} minus initial-rest median',lw=.8)
        visible_motion.extend((v-offset)[(mt>=-1)&(mt<=end+1)])
    lo,hi=min(visible_motion),max(visible_motion)
    pad=max(10,(hi-lo)*.08)
    axes[3].set_ylim(lo-pad,hi+pad)
    axes[3].set_ylabel('Acceleration change (counts)')
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=.2)
axes[0].set_title('EXP-004: orange C = clench; green T = loose-hand tilt; numbers = repetition')
axes[-1].set_xlabel('Host packet arrival time relative to first RELAX cue (seconds)')
fig.suptitle('Cue-aligned raw data — cues are instructions, not measured muscle onset')
out=Path('analysis')/p.stem.removesuffix('_raw');out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'cue_aligned_raw.png',dpi=150)
with (out/'arrival_aligned_labels.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['sample_index','host_arrival_relative_to_protocol_s','instructed_action'])
    w.writerows((r['sample_index'],t[i],labels[i]) for i,r in enumerate(rows))
steps=[(int(b['counter'])-int(a['counter']))%256 for a,b in zip(rows,rows[1:])]
summary={'samples_per_channel':len(rows),'nonunit_counter_transitions':sum(s!=1 for s in steps),
         'cue_status':cues['status'],'clench_cues':sum(e['action']=='GENTLE CLENCH' for e in events),
         'tilt_cues':sum(e['action']=='GENTLE FOREARM TILT' for e in events),
         'note':'Labels use packet arrival time; all frames in a packet share a timestamp. Screen polling and BLE transport add timing uncertainty.'}
(out/'alignment_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
