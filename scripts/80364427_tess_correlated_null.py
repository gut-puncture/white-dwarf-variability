"""Block-sign null for the independently selected ZTF frequencies in TESS.

Blocks preserve short-timescale residual structure; null symmetry and block
independence are assumptions, not a universal calibration of artifacts.
"""
from pathlib import Path
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];SID=80364427

def main():
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];freqs=[r['frequency'] for r in frozen['selected']]
    d=pd.read_csv(ROOT/f'data/{SID}_tess_s72_clean.csv.gz');t=d.time.to_numpy();y=d.PDCSAP_FLUX.to_numpy();e=d.PDCSAP_FLUX_ERR.to_numpy()
    _,ix=np.unique(np.floor(t/.2).astype(int),return_inverse=True);w=1/e**2;ws=np.bincount(ix,w)
    def project(v):return v-(np.bincount(ix,w*v)/ws)[ix]
    yy=project(y);scale=max(1.,np.sum(yy**2*w)/(len(y)-len(ws)));wy=yy*np.sqrt(w/scale);models=[]
    for f in freqs:
        phi=2*np.pi*f*(t-tref);atten=np.sinc(f*120/86400);Z=np.array([project(np.sin(phi)*atten),project(np.cos(phi)*atten)]).T*np.sqrt(w/scale)[:,None]
        inv=np.linalg.inv(Z.T@Z);score=Z.T@wy;models.append((Z,inv,float(score@inv@score)))
    out=[];rng=np.random.default_rng(20261002);draws=100000
    for width in (.2,.5,1.):
        _,group=np.unique(np.floor(t/width).astype(int),return_inverse=True);ng=int(group.max()+1)
        G=[np.array([np.bincount(group,Z[:,j]*wy,minlength=ng) for j in range(2)]).T for Z,inv,obs in models]
        exceed=0;primary_exceed=0;observed=models[0][2]
        for start in range(0,draws,1000):
            signs=rng.choice([-1.,1.],size=(min(1000,draws-start),ng));maximum=np.zeros(len(signs))
            for rank,(g,(_,inv,obs)) in enumerate(zip(G,models)):
                score=signs@g;stat=np.einsum('ni,ij,nj->n',score,inv,score);maximum=np.maximum(maximum,stat)
                if rank==0:primary_exceed+=int(np.sum(stat>=observed))
            exceed+=int(np.sum(maximum>=observed))
        row=dict(block_days=width,n_blocks=ng,observed_primary=observed,draws=draws,max20_exceedances=exceed,max20_plus_one_tail=(exceed+1)/(draws+1),primary_only_exceedances=primary_exceed,primary_only_plus_one_tail=(primary_exceed+1)/(draws+1));out.append(row);print(row,flush=True)
    result=dict(sdss_id=SID,sector=72,selection='20 ZTF discovery-frozen frequencies; TESS quality0 only; .2day offsets; no photometric clipping or aperture optimization.',tests=out,
        limitations=['Sign symmetry and independence between blocks are unverified assumptions.',
        'Signal remains in null residuals, so this test can be conservative.',
        'Three block sizes are sensitivity checks; report all rather than selecting the smallest tail.',
        'This is independent of ZTF photons but not of other TESS aperture or PDC/SAP tests.'])
    (ROOT/f'data/{SID}_tess_correlated_null.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
