"""Exploratory nearest-centroid baseline on prospectively separated whole blocks.
No thresholds or parameters selected using validation/test. No device control.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'analysis/exp004_mixed01'
RAW=ROOT/'recordings/exp004_mixed01_raw.csv'
CLASSES=['clench','rest','tilt']


def fit_centroids(features, labels):
    mean=features.mean(axis=0)
    scale=features.std(axis=0)
    scale[scale<1e-8]=1
    normalized=(features-mean)/scale
    centers=np.array([normalized[labels==label].mean(axis=0) for label in CLASSES])
    return mean,scale,centers


def predict(features,model):
    mean,scale,centers=model
    distance=np.sum((((features-mean)/scale)[:,None,:]-centers[None,:,:])**2,axis=2)
    return np.array(CLASSES)[np.argmin(distance,axis=1)]


def main():
    out=BASE/'baseline';out.mkdir(exist_ok=True)
    with RAW.open() as f: rows=list(csv.DictReader(f))
    with RAW.with_suffix('.imu.csv').open() as f: motion=list(csv.DictReader(f))
    cues=json.loads(RAW.with_suffix('.cues.json').read_text())['events']
    origin=next(e['host_monotonic_ns'] for e in cues if e['action']=='RELAX')
    t=np.array([(int(r['host_monotonic_ns'])-origin)/1e9 for r in rows])
    x=np.array([[int(r[f'ch{c}_adc']) for c in (1,2,3)] for r in rows])
    mt=np.array([(int(r['host_monotonic_ns'])-origin)/1e9 for r in motion])
    mx=np.array([[int(r[f'accel_{c}_counts']) for c in 'xyz'] for r in motion])
    intervals=json.loads((BASE/'partitioned_cue_intervals.json').read_text())
    plan=json.loads((BASE/'planned_split.json').read_text())['repetition_partition']
    features=[];metadata=[];used={};selected_intervals=[]
    for interval_id,e in enumerate(intervals):
        if e['partition']=='descriptive_only':continue
        assert e['partition']==plan[str(e['repetition'])]
        action=e['action'];label={'GENTLE CLENCH':'clench','GENTLE FOREARM TILT':'tilt','RELAX':'rest'}[action]
        # Fixed margins inherited from preceding long-hold analysis.
        start=e['start_s']+(3 if label=='rest' else 1)
        end=e['start_s']+5.5 if label=='rest' else e['end_s']-.5
        indices=np.flatnonzero((t>=start)&(t<end))
        selected_intervals.append({**e,'label':label,'feature_interval_start_s':start,'feature_interval_end_s':end})
        for offset in range(0,len(indices)-249,250):
            ix=indices[offset:offset+250]
            assert np.all(np.diff(ix)==1)
            assert not any(int(i) in used for i in ix)
            for i in ix:used[int(i)]=e['partition']
            block=x[ix].astype(float);centered=block-block.mean(axis=0)
            rms=np.sqrt(np.mean(centered**2,axis=0))
            mad=1.4826*np.median(np.abs(block-np.median(block,axis=0)),axis=0)
            wl=np.mean(np.abs(np.diff(block,axis=0)),axis=0)
            emg=np.log1p(np.concatenate([rms,mad,wl]))
            # Summarise actual independent IMU arrivals, never repeat each IMU frame at 1 kHz.
            m=mx[(mt>=t[ix[0]])&(mt<t[ix[-1]]+.001)].astype(float)
            if len(m)<2:raise ValueError('Insufficient acceleration samples for matched host interval')
            features.append(np.concatenate([emg,m.mean(axis=0),m.std(axis=0)]))
            metadata.append({'partition':e['partition'],'label':label,'repetition':e['repetition'],
                             'interval_id':interval_id,'first_emg_sample':int(ix[0]),'last_emg_sample':int(ix[-1]),
                             'host_start_s':float(t[ix[0]]),'host_end_s':float(t[ix[-1]]),'motion_samples':len(m)})
    f=np.array(features);labels=np.array([m['label'] for m in metadata]);parts=np.array([m['partition'] for m in metadata])
    report={'source_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
            'class_order':CLASSES,'window_samples':250,'window_overlap':0,
            'features':['log1p RMS per EMG channel','log1p scaled MAD per EMG channel','log1p mean absolute first difference per EMG channel',
                        'Fused only: acceleration X/Y/Z mean and standard deviation from host-matched interval'],
            'partition_window_counts':{p:{c:int(np.sum((parts==p)&(labels==c))) for c in CLASSES} for p in ('train','validation','test')},
            'models':{},'limitations':['Exploratory, single-session evaluation; hold-only and late-instructed-rest windows.',
              'One paired clench/tilt block in validation and one in test; windows are not independent trials.',
              'Cue-relative labels do not guarantee settled physiology; carryover can affect rest and tilt.',
              'Not a continuous detector: transitions, onset/release latency, unknown gestures and false commands/minute not evaluated.',
              'QC plots of all partitions were viewed before this baseline; not a pristine blinded confirmatory test.',
              'No parameter tuning or model selection on validation/test. Later independent session required.']}
    predictions=[]
    fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
    for row,(name,n_features) in enumerate([('emg_only',9),('emg_and_acceleration',15)]):
        model=fit_centroids(f[parts=='train',:n_features],labels[parts=='train'])
        model_result={}
        model_file={'feature_count':n_features,'classes':CLASSES,
                    'mean':model[0].tolist(),'scale':model[1].tolist(),'centroids':model[2].tolist(),
                    'method':'Nearest class centroid after training-only per-feature standardization; no probability/confidence output.'}
        (out/f'{name}_model.json').write_text(json.dumps(model_file,indent=2))
        for col,part in enumerate(('validation','test')):
            ix=np.flatnonzero(parts==part);pred=predict(f[ix,:n_features],model)
            matrix=np.array([[np.sum((labels[ix]==a)&(pred==b)) for b in CLASSES] for a in CLASSES])
            recall=np.diag(matrix)/matrix.sum(axis=1)
            model_result[part]={'confusion_matrix':matrix.tolist(),'window_accuracy':float(np.mean(pred==labels[ix])),
                                'balanced_accuracy':float(recall.mean()),'recall':dict(zip(CLASSES,recall.tolist()))}
            for i,guess in zip(ix,pred):predictions.append({'model':name,**metadata[i],'prediction':str(guess)})
            ax=axes[row,col];ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,matrix.max()))
            for i in range(3):
                for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
            ax.set_xticks(range(3),CLASSES);ax.set_yticks(range(3),CLASSES)
            ax.set_xlabel('Predicted instruction class');ax.set_ylabel('Recorded instruction class')
            ax.set_title(f'{name.replace("_"," ")} / {part}\nBalanced accuracy {recall.mean():.1%}')
        report['models'][name]=model_result
    fig.suptitle('Exploratory baseline — whole-block split, one session, selected intervals only')
    fig.savefig(out/'confusion_matrices.png',dpi=150);plt.close(fig)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    (out/'selected_intervals.json').write_text(json.dumps(selected_intervals,indent=2))
    with (out/'window_features.csv').open('w',newline='') as target:
        w=csv.writer(target);w.writerow([*metadata[0].keys(),*[f'feature_{i}' for i in range(15)]])
        for m,v in zip(metadata,f):w.writerow([*m.values(),*v])
    with (out/'predictions.csv').open('w',newline='') as target:
        w=csv.DictWriter(target,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    print(json.dumps({'counts':report['partition_window_counts'],'results':report['models']},indent=2))


if __name__=='__main__':main()
