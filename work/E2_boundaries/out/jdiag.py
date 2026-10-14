import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
from s2_replay import detected_junctions
PKG="/home/user/data_raw/abc2027_uwb_train_v1"
ev=pd.read_csv(PKG+"/streams_events.csv"); co=pd.read_csv(PKG+"/streams_composition.csv")
P=pd.read_csv("out/train_oof_proto.csv")
fs=co[(co.kind=="fall")&co.stream_id.str.startswith("train")]
res=[]
for sid,Q in P.groupby("stream_id"):
    for q in (95,97):
        J,_=detected_junctions(sid,q)
        for r in Q.itertuples():
            seg=fs[(fs.stream_id==sid)&(fs.t_start<r.t_end)&(fs.t_end>r.t_start)]
            if len(seg)==0: continue
            tj=seg.t_start.iloc[0]+0.1  # crossfade centre
            near=np.min(np.abs(J-tj)) if len(J) else 99
            for g in (0.2,0.5,1.0):
                prev=J[J<r.t_start-g]; pj=prev[-1] if len(prev) else np.nan
                res.append(dict(q=q,g=g,found=near<0.3,ok=abs(pj-tj)<0.3,gap=r.t_start-tj))
R=pd.DataFrame(res)
print(R.groupby(['q','g'])[['found','ok']].mean())
print(R[(R.q==97)&(R.g==1.0)&R.found&~R.ok].gap.describe())
