"""Descriptive EXP-004 analysis. No host filters, classification or onset fitting.
Run from project root. Cue labels are uncertain by screen/reaction/BLE latency.
"""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'recordings/exp004_attempt03_raw.csv'
OUT=ROOT/'analysis/exp004_attempt03/quality'
FS=1000.0  # nominal; received throughput verified separately


def psd_segments(segments):
    n=500
    window=np.hanning(n+1)[:-1]  # periodic Hann
    spectra=[]
    for values in segments:
        for start in range(0,len(values)-n+1,n//2):
            block=values[start:start+n]
            block=block-block.mean(axis=0)
            power=np.abs(np.fft.rfft(block*window[:,None],axis=0))**2
            power/=FS*np.sum(window**2)
            power[1:-1]*=2
            spectra.append(power)
    return np.fft.rfftfreq(n,1/FS),np.mean(spectra,axis=0),len(spectra)


def main():
    global SOURCE, OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=SOURCE)
    args=parser.parse_args()
    SOURCE=args.input.resolve()
    OUT=ROOT/'analysis'/SOURCE.stem.removesuffix('_raw')/'quality'
    OUT.mkdir(parents=True,exist_ok=True)
    with SOURCE.open() as f: rows=list(csv.DictReader(f))
    events=json.loads(SOURCE.with_suffix('.cues.json').read_text())['events']
    origin=next(e['host_monotonic_ns'] for e in events if e['action']=='RELAX')
    duration=(next(e['host_monotonic_ns'] for e in events if e['action']=='DONE')-origin)/1e9
    arrival=np.array([(int(r['host_monotonic_ns'])-origin)/1e9 for r in rows])
    x=np.array([[int(r[f'ch{c}_adc']) for c in (1,2,3)] for r in rows],dtype=float)
    clenches=[]
    for i,e in enumerate(events):
        if e['action']=='GENTLE CLENCH':
            clenches.append(((e['host_monotonic_ns']-origin)/1e9,
                             (events[i+1]['host_monotonic_ns']-origin)/1e9))
    # Fixed margins selected before calculating metrics; never shifted to fit peaks.
    intervals={'initial_rest':[(2,9)],'clench':[(a+.5,b-.25) for a,b in clenches],
               'after_release':[(b+.25,b+1.5) for a,b in clenches],
               'late_gap':[(b+1.5,b+2.75) for a,b in clenches],
               'final_rest':[(duration-8,duration-1)]}
    if duration>60:
        intervals['clench']=[(a+1,b-.5) for a,b in clenches]
        intervals['late_gap']=[(b+3,b+5.5) for a,b in clenches]
    metrics={};blocks_by_state={};window_rows=[];segment_rows=[]
    for state,spans in intervals.items():
        values=[];rms=[];mad=[];per_interval=[]
        for rep,(a,b) in enumerate(spans,1):
            selected=x[(arrival>=a)&(arrival<b)]
            values.append(selected)
            this_rms=[]
            for offset in range(0,len(selected)-249,250):
                block=selected[offset:offset+250]
                centered=block-block.mean(axis=0)
                r=np.sqrt(np.mean(centered**2,axis=0))
                m=1.4826*np.median(np.abs(block-np.median(block,axis=0)),axis=0)
                rms.append(r);mad.append(m);this_rms.append(r)
                window_rows.append([state,rep,a+offset/FS,*r,*m])
            per_interval.append(np.median(this_rms,axis=0).tolist())
            segment_rows.append([state,rep,a,b,len(selected)])
        blocks_by_state[state]=values
        metrics[state]={'intervals_s':spans,'samples':sum(map(len,values)),
                        'windows_250_samples':len(rms),
                        'median_centered_rms_counts':np.median(rms,axis=0).tolist(),
                        'median_scaled_mad_counts':np.median(mad,axis=0).tolist(),
                        'per_interval_median_rms_counts':per_interval}
    baseline=np.array(metrics['initial_rest']['median_centered_rms_counts'])
    for state in metrics:
        metrics[state]['rms_ratio_to_initial_rest']=(np.array(metrics[state]['median_centered_rms_counts'])/baseline).tolist()
    spectra={};spectral_summary={}
    for state in ('initial_rest','clench','after_release','final_rest'):
        f,p,n=psd_segments(blocks_by_state[state]);spectra[state]=p
        bands={}
        for name,lo,hi in [('2_to_20',2,20),('20_to_100',20,100),('100_to_450',100,450)]:
            bands[name]=(np.sum(p[(f>=lo)&(f<hi)],axis=0)*(f[1]-f[0])).tolist()
        active=(f>=2)&(f<=450)
        peaks=f[active][np.argmax(p[active],axis=0)].tolist()
        spectral_summary[state]={'averaged_periodograms':n,'strongest_bin_2_to_450_hz':peaks,
                                 'band_power_counts_squared':bands}
    # Descriptive line prominence, not a mains attribution or significance test.
    p=spectra['initial_rest']
    lines={}
    for hz in (50,100,150,200):
        center=np.abs(f-hz)<=2
        nearby=(np.abs(f-hz)>=6)&(np.abs(f-hz)<=14)
        lines[str(hz)]=(np.max(p[center],axis=0)/np.median(p[nearby],axis=0)).tolist()
    motion_summary={}
    motion_path=SOURCE.with_suffix('.imu.csv')
    if motion_path.exists():
        with motion_path.open() as motion_file: motion=list(csv.DictReader(motion_file))
        mt=np.array([(int(r['host_monotonic_ns'])-origin)/1e9 for r in motion])
        mx=np.array([[int(r[f'accel_{axis}_counts']) for axis in ('x','y','z')] for r in motion],dtype=float)
        for state,spans in intervals.items():
            per_interval=[]
            for a,b in spans:
                v=mx[(mt>=a)&(mt<b)]
                per_interval.append({'samples':len(v),'median_xyz_counts':np.median(v,axis=0).tolist(),
                    'centered_rms_xyz_counts':np.sqrt(np.mean((v-v.mean(axis=0))**2,axis=0)).tolist()})
            motion_summary[state]=per_interval
    report={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'protocol_host_seconds':duration,'motion_intervals':motion_summary,
            'nominal_sps':FS,'metrics':metrics,'spectra':spectral_summary,
            'initial_rest_line_peak_to_local_median':lines,
            'method':'Nonoverlapping 250-sample windows; subtract window mean for RMS; '
                     'scaled MAD=1.4826*median absolute deviation from window median. '
                     'PSD: 500-sample periodic-Hann, 50% overlap, per-window mean removal, '
                     'one-sided density; average within state without crossing interval boundaries.',
            'limitations':['One session, five repetitions; no statistical generalisation or SNR estimate.',
                           'Fixed cue-relative margins are assumptions, not verified muscle onset.',
                           'Arrival-time labels retain screen/reaction/BLE uncertainty.',
                           'Spectral bands are descriptive; analogue bandwidth/calibration unverified.']}
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2))
    with (OUT/'window_metrics.csv').open('w',newline='') as out:
        w=csv.writer(out);w.writerow(['state','interval','approx_window_start_s','rms_ch1','rms_ch2','rms_ch3','mad_ch1','mad_ch2','mad_ch3']);w.writerows(window_rows)
    with (OUT/'intervals.csv').open('w',newline='') as out:
        w=csv.writer(out);w.writerow(['state','interval','start_s','end_s','samples']);w.writerows(segment_rows)
    fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    colors={'initial_rest':'#52677c','clench':'#b35b0a','after_release':'#8c3c91','late_gap':'#2c7e59','final_rest':'#2469a0'}
    pretty={'initial_rest':'Initial rest','clench':'Clench interior','after_release':'After release','late_gap':'Late gap','final_rest':'Final rest'}
    for c in range(3):
        # Full-protocol rolling display uses disjoint quarter-second sample blocks.
        keep=(arrival>=0)&(arrival<duration);v=x[keep];times=arrival[keep]
        tt=[];rr=[]
        for j in range(0,len(v)-249,250):
            z=v[j:j+250,c];tt.append(np.mean(times[j:j+250]));rr.append(np.sqrt(np.mean((z-z.mean())**2)))
        axes[c,0].plot(tt,rr,color='#2469a0',lw=1)
        for a,b in clenches: axes[c,0].axvspan(a,b,color='#e6a23c',alpha=.25)
        axes[c,0].set_ylabel(f'CH{c+1} centered RMS (counts)');axes[c,0].set_xlim(0,duration)
        for state,p in spectra.items():
            axes[c,1].semilogy(f[1:],p[1:,c],label=pretty[state],color=colors[state],lw=1)
        axes[c,1].set_xlim(2,450);axes[c,1].set_ylabel(f'CH{c+1} PSD (counts²/Hz)')
        for ax in axes[c]:ax.grid(alpha=.2)
    axes[0,0].set_title('250 ms variation; shading = instructed clench')
    axes[0,1].set_title('Spectra from fixed cue-relative intervals')
    axes[0,1].legend(fontsize=8)
    axes[-1,0].set_xlabel('Host arrival time from first RELAX cue (s)')
    axes[-1,1].set_xlabel('Frequency (Hz; nominal 1000 SPS)')
    fig.savefig(OUT/'variation_and_spectra.png',dpi=150);plt.close(fig)
    print(json.dumps({'rms':{k:v['median_centered_rms_counts'] for k,v in metrics.items()},
                      'ratios':{k:v['rms_ratio_to_initial_rest'] for k,v in metrics.items()},
                      'peaks':{k:v['strongest_bin_2_to_450_hz'] for k,v in spectral_summary.items()},
                      'line_prominence':lines},indent=2))


if __name__=='__main__': main()
