"""Evaluate frozen nearest-centroid models on a new mixed acquisition; no fit calls."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/exp004_frozen01'
RAW=ROOT/'recordings/exp004_frozen01_raw.csv'


def features(block,motion):
    block=block.astype(float)
    rms=np.sqrt(np.mean((block-block.mean(axis=0))**2,axis=0))
    mad=1.4826*np.median(np.abs(block-np.median(block,axis=0)),axis=0)
    wl=np.mean(np.abs(np.diff(block,axis=0)),axis=0)
    return np.concatenate([np.log1p(np.concatenate([rms,mad,wl])),motion.mean(axis=0),motion.std(axis=0)])


def main():
    plan=json.loads((OUT/'evaluation_plan.json').read_text())
    models={}
    for filename,digest in plan['models_sha256'].items():
        p=OUT/'frozen_models'/filename
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,'Frozen model changed'
        models[filename.removesuffix('_model.json')]=json.loads(p.read_text())
    with RAW.open() as f:rows=list(csv.DictReader(f))
    with RAW.with_suffix('.imu.csv').open() as f:motion=list(csv.DictReader(f))
    cue=json.loads(RAW.with_suffix('.cues.json').read_text())
    assert cue['status']=='complete'
    events=cue['events'];origin=next(e['host_monotonic_ns'] for e in events if e['action']=='RELAX')
    end=(events[-1]['host_monotonic_ns']-origin)/1e9
    t=np.array([(int(r['host_monotonic_ns'])-origin)/1e9 for r in rows])
    x=np.array([[int(r[f'ch{c}_adc']) for c in (1,2,3)] for r in rows])
    mt=np.array([(int(r['host_monotonic_ns'])-origin)/1e9 for r in motion])
    mx=np.array([[int(r[f'accel_{c}_counts']) for c in 'xyz'] for r in motion],dtype=float)
    windows=[];vectors=[]
    for i,e in enumerate(events[:-1]):
        if not e['repetition']:continue
        start=(e['host_monotonic_ns']-origin)/1e9
        finish=(events[i+1]['host_monotonic_ns']-origin)/1e9
        label={'GENTLE CLENCH':'clench','GENTLE FOREARM TILT':'tilt','RELAX':'rest'}[e['action']]
        a=start+(3 if label=='rest' else 1);b=start+5.5 if label=='rest' else finish-.5
        ix=np.flatnonzero((t>=a)&(t<b))
        for offset in range(0,len(ix)-249,250):
            ids=ix[offset:offset+250];assert np.all(np.diff(ids)==1)
            m=mx[(mt>=t[ids[0]])&(mt<t[ids[-1]]+.001)]
            assert len(m)>=2
            vectors.append(features(x[ids],m))
            windows.append({'label':label,'repetition':e['repetition'],'interval_id':i,
                            'first_emg_sample':int(ids[0]),'last_emg_sample':int(ids[-1]),
                            'host_start_s':float(t[ids[0]]),'host_end_s':float(t[ids[-1]])})
    vectors=np.array(vectors);assert np.isfinite(vectors).all()
    labels=np.array([w['label'] for w in windows]);predictions=[];results={}
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    for ax,(name,m) in zip(axes,models.items()):
        n=m['feature_count'];mean=np.array(m['mean']);scale=np.array(m['scale']);centers=np.array(m['centroids']);classes=m['classes']
        distance=np.sum((((vectors[:,:n]-mean)/scale)[:,None,:]-centers[None,:,:])**2,axis=2)
        pred=np.array(classes)[np.argmin(distance,axis=1)]
        matrix=np.array([[np.sum((labels==a)&(pred==b)) for b in classes] for a in classes])
        recall=np.diag(matrix)/matrix.sum(axis=1)
        per_rep={}
        for rep in (1,2,3,4):
            selected=np.array([w['repetition']==rep for w in windows]);per_rep[str(rep)]=float(np.mean(pred[selected]==labels[selected]))
        results[name]={'class_order':classes,'windows':len(windows),'window_accuracy':float(np.mean(pred==labels)),
                       'balanced_accuracy':float(recall.mean()),'recall':dict(zip(classes,recall.tolist())),
                       'confusion_matrix':matrix.tolist(),'per_paired_block_window_accuracy':per_rep}
        for w,guess in zip(windows,pred):predictions.append({'model':name,**w,'prediction':str(guess)})
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,matrix.max()))
        for i in range(3):
            for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        ax.set_xticks(range(3),classes);ax.set_yticks(range(3),classes);ax.set_xlabel('Predicted class');ax.set_ylabel('Instructed class')
        ax.set_title(f'{name.replace("_"," ")}\nAccuracy {np.mean(pred==labels):.1%}; balanced {recall.mean():.1%}')
    fig.suptitle('Frozen models — new acquisition; selected hold / late-rest windows')
    fig.savefig(OUT/'frozen_confusion_matrices.png',dpi=150);plt.close(fig)
    with (OUT/'frozen_predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    report={'source_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),'models_sha256':plan['models_sha256'],
            'protocol_host_seconds':end,'class_counts':{c:int(np.sum(labels==c)) for c in ('clench','rest','tilt')},
            'results':results,'limitations':['No refitting/recalibration; all new selected windows evaluated.',
              'Same-day new acquisition, removal/re-wear and separate-session status not established.',
              'Instruction labels and host timing uncertain; neighbouring windows are correlated.',
              'Transitions, ordinary activities, continuous false commands and onset latency not evaluated.',
              'No rejection threshold; forced choice among three classes.']}
    (OUT/'frozen_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
