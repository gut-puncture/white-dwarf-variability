"""Fixed-period image attribution, including covariance of the close pair."""
import os,json,importlib,itertools
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SID=65258778
PAIR=['1974721783974773376','1974720310808109696']


def nuisance(d):
    season=np.floor((d.mjd.to_numpy()-58000)/365.25).astype(int)
    B=np.array([(season==s).astype(float) for s in np.unique(season)]).T
    for s in np.unique(season):
        m=season==s;t=d.time.to_numpy();B=np.c_[B,m*(t-np.mean(t[m]))/100]
    for name in ['seeing','airmass','noise_pixel','magzp']:
        v=d[name].to_numpy();v=(v-np.mean(v))/max(np.std(v),1e-9);B=np.c_[B,v]
        if name=='seeing':B=np.c_[B,v*v]
    return B


def scalar_fit(d,f,tref):
    y=d.flux_zp25.to_numpy();e=d.error_zp25.to_numpy();B=nuisance(d)
    coef=np.linalg.lstsq(B/e[:,None],y/e,rcond=None)[0];r=y-B@coef;dof=len(y)-np.linalg.matrix_rank(B)
    fun=lambda j:np.sum(r*r/(e*e+j*j))-dof
    jitter=brentq(fun,0,1e5) if fun(0)>0 else 0.;e=np.hypot(e,jitter)
    u,s,_=np.linalg.svd(B/e[:,None],full_matrices=False);Q=u[:,s>s[0]*1e-10]
    def project(z):return z-Q@(Q.T@z)
    phase=2*np.pi*f*(d.time.to_numpy()-tref);H=np.c_[np.sin(phase),np.cos(phase)]/e[:,None]
    X=project(H);v=project(y/e);b=np.linalg.lstsq(X,v,rcond=None)[0];cov=np.linalg.inv(X.T@X);res=v-X@b
    nights=np.floor(d.mjd.to_numpy()).astype(int);groups=np.unique(nights)
    scores=np.array([X[nights==g].T@res[nights==g] for g in groups]);cluster=cov@(scores.T@scores)@cov*len(groups)/(len(groups)-1)
    result=dict(n=len(y),n_nights=len(groups),nuisance_rank=Q.shape[1],jitter_flux=jitter,delta=float(v@v-res@res),coefficients=b.tolist(),covariance=cov.tolist(),night_cluster_covariance=cluster.tolist(),amplitude=float(np.linalg.norm(b)),phase=float(np.arctan2(b[1],b[0])))
    return result,dict(y=v,X=X,error=e,night=nights)


def pair_models(d,records,mode,f,tref):
    frames=[];ys=[];cs=[]
    for key,g in d.groupby('filefracday',sort=False):
        p=g.set_index('source_id')
        if not all(s in p.index for s in PAIR):continue
        r=records.get(str(key))
        if not r or 'pair_covariances' not in r:continue
        frames.append(g.iloc[0]);ys.append(p.loc[PAIR].flux_zp25.to_numpy());cs.append(r['pair_covariances'][mode])
    m=pd.DataFrame(frames).reset_index(drop=True);y=np.asarray(ys);C=np.asarray(cs);B=nuisance(m);n=len(m);p=B.shape[1]
    X0=np.zeros((n,2,2*p));X0[:,0,:p]=B;X0[:,1,p:]=B
    L=np.linalg.cholesky(C);white=np.linalg.inv(L)
    z=np.einsum('nij,nj->ni',white,y).ravel();N=np.einsum('nij,njk->nik',white,X0).reshape(2*n,2*p)
    u,s,_=np.linalg.svd(N,full_matrices=False);Q=u[:,s>s[0]*1e-10]
    def project(a):return a-Q@(Q.T@a)
    v=project(z);scale=max(1,float(v@v)/(len(v)-Q.shape[1]));v/=np.sqrt(scale)
    phi=2*np.pi*f*(m.time.to_numpy()-tref);H=np.c_[np.sin(phi),np.cos(phi)]
    designs=[];models=[]
    for which in ['white_dwarf','neighbour','both']:
        ix=[0] if which=='white_dwarf' else [1] if which=='neighbour' else [0,1]
        raw=np.zeros((n,2,2*len(ix)))
        for k,i in enumerate(ix):raw[:,i,2*k:2*k+2]=H
        X=np.einsum('nij,njk->nik',white,raw).reshape(2*n,2*len(ix));X=project(X)/np.sqrt(scale)
        coef=np.linalg.lstsq(X,v,rcond=None)[0];res=v-X@coef
        models.append(dict(hypothesis=which,delta=float(v@v-res@res),coefficients=coef.tolist(),covariance=np.linalg.inv(X.T@X).tolist()))
        designs.append(X)
    return dict(n_images=n,null_variance_inflation=scale,models=models,white_dwarf_minus_neighbour_fit_gain=models[0]['delta']-models[1]['delta'],limitations='Pair covariance propagated from deblending; one empirical null scale. Temporal calibration terms profiled separately per source. Not an independent detection from catalogue photons.')


def main():
    plan=pd.read_csv(ROOT/f'data/{SID}_ztf_image_plan.csv',dtype={'filefracday':str})
    d=pd.read_csv(ROOT/f'data/{SID}_ztf_image_photometry.csv',dtype={'source_id':str,'filefracday':str})
    d=d[d.usable&d.error_zp25.gt(0)&np.isfinite(d.flux_zp25)].merge(plan[['filefracday','airmass']],on='filefracday',validate='many_to_one')
    fz=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());f=fz['selected'][0]['frequency'];tref=fz['tref']
    records={p.stem:json.loads(p.read_text()) for p in (ROOT/f'data/{SID}_ztf_image_measurements').glob('*.json') if 'superseded' not in p.name}
    results=[];pair=[];injections=[]
    for (mode,source),g in d.groupby(['mode','source_id']):
        if len(g)<40:continue
        r,detail=scalar_fit(g,f,tref);r.update(mode=mode,source_id=source,subset='full');results.append(r)
        if source in PAIR:
            train=g[g.split=='discovery'];valid=g[g.split=='validation']
            if len(train)>=40 and len(valid)>=25:
                a,_=scalar_fit(train,f,tref);b,v=scalar_fit(valid,f,tref)
                prediction=v['X']@np.asarray(a['coefficients']);gain=float(v['y']@v['y']-np.sum((v['y']-prediction)**2))
                results.append(dict(mode=mode,source_id=source,subset='chronological',training=a,validation=b,unchanged_prediction_gain=gain))
    for mode,g in d[d.source_id.isin(PAIR)].groupby('mode'):
        if g.filefracday.nunique()<40:continue
        pair.append(dict(mode=mode,**pair_models(g,records,mode,f,tref)))
        for true_index,true_source in enumerate(PAIR):
            for trial_id in range(7):
                matrices=[];rows=[]
                for key,part in g.groupby('filefracday'):
                    r=records.get(str(key));trials=[x for x in r.get('injection',[]) if x['mode']==mode] if r else []
                    if len(trials)!=7:continue
                    q=part.set_index('source_id')
                    if not all(s in q.index for s in PAIR):continue
                    matrices.append(np.asarray(trials[trial_id]['response'])[:,true_index]);rows.append(q.loc[PAIR].reset_index())
                if not rows:continue
                response=np.asarray(matrices)
                # Propagate an actual frozen-period injected waveform through
                # each exposure's deblending response and the time regression.
                recovered=[]
                for recovered_index,source in enumerate(PAIR):
                    sample=pd.DataFrame([r[r.source_id==source].iloc[0] for r in rows]).reset_index(drop=True)
                    actual,detail=scalar_fit(sample,f,tref)
                    phase=2*np.pi*f*(sample.time.to_numpy()-tref)
                    b=np.array(fz['selected'][0]['coefficients'][:2]);b=b/np.linalg.norm(b)
                    wave=b[0]*np.sin(phase)+b[1]*np.cos(phase)
                    sample['flux_zp25']=wave*response[:,recovered_index]
                    sample['error_zp25']=detail['error']
                    measurement,_=scalar_fit(sample,f,tref)
                    recovered.append(dict(source_id=source,recovered_unit_amplitude=measurement['amplitude'],projection_on_injected_phase=float(np.asarray(measurement['coefficients'])@b)))
                par={k:v for k,v in trials[trial_id].items() if k not in ['response','source_order']}
                injections.append(dict(mode=mode,true_source=true_source,trial=par,n_images=len(rows),median_response=np.median(response,axis=0).tolist(),response_percentile_5_95=np.percentile(response,[5,95],axis=0).tolist(),temporal_injection_recovery=recovered))
    out=dict(frequency=f,n_planned_images=len(plan),n_extracted_images=int(d.filefracday.nunique()),complete_plan=bool(d.filefracday.nunique()==len(plan)),scalar_results=results,pair_comparisons=pair,injection_response=injections,
        limitations=['Same photons as original catalogue; image attribution is not another independent discovery.',
        'Only g-band field770 images; missing products are retained in acquisition manifest.',
        'Background covariance, PSF variation across detector, correlated reference noise and chromatic subtraction are imperfectly modeled.',
        'Synthetic PSFs vary width by10percent and centroid by0.15pixel; this bounds specified perturbations, not all image systematics.',
        'Period fixed before images; separate source fits and derivative variants are correlated diagnostics.'])
    (ROOT/f'data/{SID}_ztf_image_period.json').write_text(json.dumps(out,indent=2)+'\n')
    print('IMAGES',out['n_extracted_images'],'/',len(plan),flush=True)
    for r in results:
        if r['source_id'] in PAIR:print('SOURCE',json.dumps(r),flush=True)
    for r in pair:print('PAIR',json.dumps(r),flush=True)


if __name__=='__main__':main()
