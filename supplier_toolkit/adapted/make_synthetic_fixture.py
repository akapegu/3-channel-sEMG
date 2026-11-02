"""Generate labelled synthetic files solely for training-path smoke testing."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from rate_filters import PROFILE
root=Path(__file__).resolve().parents[1]/'verification'/'synthetic_data'
root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(91);manifest=[]
for trial,part in enumerate(('train','validation','test'),1):
 for c,label in enumerate(('extension','flexion','pinch','rest')):
  folder=root/label;folder.mkdir(exist_ok=True)
  path=folder/f'{label}_trial{trial}_synthetic.csv'
  t=np.arange(2000)/1000
  v=rng.normal(size=(2000,6));v[:,:3]+=np.sin(2*np.pi*(80+c*30)*t[:,None])*(c+1)
  v[:,3:]+=c*10
  pd.DataFrame(v,columns=['ch1','ch2','ch3','accel_x','accel_y','accel_z']).to_csv(path,index=False)
  path.with_suffix('.beep.json').write_text(json.dumps({'segments':[[0,2000]]}))
  path.with_suffix('.processing.json').write_text(json.dumps({'sampling_rate':1000,'preprocessing_profile':PROFILE,'synthetic_only':True}))
  manifest.append({'path':str(path.relative_to(root)),'partition':part,'acquisition_group':f'synthetic_acquisition_{trial}'})
(root/'split.json').write_text(json.dumps({'files':manifest},indent=2))
print(root)
