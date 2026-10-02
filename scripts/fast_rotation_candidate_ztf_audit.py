"""Detailed robustness checks of an already selected, frozen-frequency lead."""
from pathlib import Path
import json,argparse
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar
import fast_rotation_search as search
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--sid',type=int,required=True);p.add_argument('--null-draws',type=int,default=10000);p.add_argument('--family-size',type=int,default=44);a=p.parse_args();sid=a.sid
    d=pd.read_csv(ROOT/f'data/fast_rotation/{sid}_prepared.csv.gz',dtype={'oid':str})
    frozen=json.loads((ROOT/f'data/fast_rotation/{sid}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];f=frozen['selected'][0]['frequency']
    rows=[];full=search.SplitModel(d,tref);step=1/(np.ptp(full.t)*5)
    for name,part,quality,jitter in [('all_quality_jitter',d,True,True),('all_no_quality_jitter',d,False,True),('all_quality_no_jitter',d,True,False),
        ('30second_only',d[d.exptime==30],True,True),('green',d[d.filtercode=='zg'],True,True),('red',d[d.filtercode=='zr'],True,True)]:
        m=search.SplitModel(part,tref,systematics=quality,jitter=jitter);r=m.fit(f,True);r['case']=name;rows.append(r)
    aliases=[]
    for r in frozen['selected']:
        f0=r['frequency'];opt=minimize_scalar(lambda off:full.fit(f0+off),bounds=(-2*step,2*step),method='bounded',options={'xatol':1e-11})
        fit=full.fit(f0+opt.x,True);fit['discovery_frequency']=f0;aliases.append(fit)
    aliases.sort(key=lambda r:r['chi2'])
    seasons=[]
    for season,part in d.groupby(np.floor((d.mjd-58000)/365.25).astype(int)):
        if len(part)<40:continue
        for b,band in part.groupby('filtercode'):
            if len(band)<20:continue
            m=search.SplitModel(band,tref);r=m.fit(f,True);r.update(season=int(season),median_mjd=float(band.mjd.median()),band=b);seasons.append(r)
    # The only frequency selection here is the prospectively frozen discovery
    # list. Null draws sign-flip entire validation nights jointly across bands.
    v=search.SplitModel(d[d.split=='validation'],tref)
    bases=[]
    for r in frozen['selected']:
        z=v.project(v.design(r['frequency']));u,s,_=np.linalg.svd(z,full_matrices=False);rank=int(sum(s>s[0]*1e-10));assert rank==2*len(v.bands);bases.append(u[:,:rank])
    basis=np.hstack(bases);rank=2*len(v.bands);_,night=np.unique(np.floor(v.d.mjd),return_inverse=True)
    rng=np.random.default_rng(20261002);maxima=[]
    for start in range(0,a.null_draws,500):
        n=min(500,a.null_draws-start);signs=rng.choice([-1.,1.],size=(int(night.max()+1),n));sim=v.ry[:,None]*signs[night]
        values=basis.T@sim;gains=np.sum(values.reshape(len(bases),rank,n)**2,axis=1);maxima.extend(gains.max(axis=0).tolist())
    observed=v.fit(f,True)['delta_chi2'];k=int(np.sum(np.array(maxima)>=observed))
    null=dict(method='Night-sign wild null on validation residuals, max over20 discovery-frozen frequencies',draws=a.null_draws,seed=20261002,
        observed_at_discovery_maximum=observed,exceedances=k,plus_one_tail=(k+1)/(a.null_draws+1),family_size=a.family_size,family_adjusted_plus_one_tail=min(1.,a.family_size*(k+1)/(a.null_draws+1)),
        maxima=maxima,limitations='Null symmetry and inter-night independence are assumptions. Signal remains in null residuals; this is conservative but not an all-artifact calibration.')
    out=dict(sdss_id=sid,discovery_frequency=f,full_data_robustness=rows,full_data_alias_refits=aliases,seasons=seasons,validation_night_null=null)
    (ROOT/f'data/{sid}_fast_rotation_ztf_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,axs=plt.subplots(2,2,figsize=(11,8));colours={'discovery':'#416f9c','validation':'#e48839'}
    for j,b in enumerate(['zg','zr']):
        for split in ['discovery','validation']:
            part=d[(d.filtercode==b)&(d.split==split)];m=search.SplitModel(part,tref);r=m.fit(f,True);phase=(f*m.t)%1
            for k in range(12):
                ix=np.floor(phase*12)==k
                if ix.sum():w=1/m.error[ix]**2;axs[0,j].errorbar((k+.5)/12,np.average(m.residual[ix],weights=w),1/np.sqrt(sum(w)),fmt='o',color=colours[split],label=split if k==0 else None)
            ph=np.linspace(0,1,200);a1,b1=r['coefficients'];axs[0,j].plot(ph,a1*np.sin(2*np.pi*ph)+b1*np.cos(2*np.pi*ph),color=colours[split],lw=1)
        axs[0,j].invert_yaxis();axs[0,j].set(xlabel='Phase',ylabel='Corrected magnitude residual',title=b+' at discovery-only period');axs[0,j].legend()
        for s in seasons:
            if s['band']!=b:continue
            p=s['parameters'][0];axs[1,j].errorbar(s['median_mjd'],p['phase_radians'],p['phase_error_radians'],fmt='o',color='#416f9c')
        axs[1,j].set(xlabel='MJD',ylabel='Fitted magnitude phase (radians)',title=b+' phase across seasons')
    fig.suptitle(f'{sid}: {86400/f:.6f}-second ZTF candidate');fig.tight_layout();fig.savefig(ROOT/f'figures/{sid}_fast_rotation_ztf.png',dpi=160);plt.close(fig)
    print('Null:',{k:v for k,v in null.items() if k!='maxima'},flush=True)
    print('Top aliases',[(r['period_seconds'],r['delta_chi2']) for r in aliases[:5]],flush=True)

if __name__=='__main__':main()
