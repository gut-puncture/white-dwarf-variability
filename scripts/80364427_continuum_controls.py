"""Check the target's fixed-window colour against same-exposure comparison stars."""
from pathlib import Path
import os,json,importlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from astropy.io import fits
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
window=importlib.import_module('80364427_continuum_windows')
ROOT=window.ROOT;SID=80364427

def matrices(d,f,tref,mode):
    y=d.value.to_numpy().copy();e=d.error.to_numpy();groups=d.file.to_numpy()
    B=np.array([(groups==x).astype(float) for x in np.unique(groups)]).T
    if mode=='subtract':y-=d.common_mode.to_numpy()
    if mode=='fit':
        for field in np.unique(d.fieldid):B=np.c_[B,d.common_mode.to_numpy()*(d.fieldid.to_numpy()==field)]
    N=B/e[:,None];u,s,_=np.linalg.svd(N,full_matrices=False);rank=sum(s>max(s)*1e-10);Q=u[:,:rank]
    if len(y)-rank<4:raise ValueError('Fewer than4 degrees of freedom remain after calibration nuisance terms')
    def project(a):return a-Q@(Q.T@a)
    wy=project(y/e);null=float(wy@wy);scale=max(1,null/max(1,len(y)-rank));wy/=np.sqrt(scale)
    phi=2*np.pi*f*(d.time.to_numpy()-tref);att=np.sinc(f*d.exptime.to_numpy()/86400)
    X=project(att[:,None]*np.c_[np.sin(phi),np.cos(phi)]/e[:,None])/np.sqrt(scale)
    singular=np.linalg.svd(X,compute_uv=False)
    if singular[-1]<singular[0]*1e-8:raise ValueError('Unidentifiable sinusoid')
    return wy,X,Q,scale,rank

def fit(d,f,tref,mode,coefficients=None):
    y,X,Q,scale,rank=matrices(d,f,tref,mode);cov=np.linalg.inv(X.T@X)
    b=cov@(X.T@y) if coefficients is None else np.asarray(coefficients);res=y-X@b
    null=float(y@y);chi=float(res@res);amp=float(np.linalg.norm(b))
    return dict(n=len(d),n_nights=int(d.mjd.nunique()),nuisance_rank=int(rank),variance_inflation=scale,
        null_chi2=null,chi2=chi,delta_chi2=null-chi,coefficients=b.tolist(),covariance=cov.tolist(),amplitude_mag=amp,
        amplitude_error=float(np.sqrt(b@cov@b)/max(amp,1e-20)),phase_radians=float(np.arctan2(b[1],b[0])))

def main():
    plan=pd.read_csv(ROOT/f'data/{SID}_spectral_control_plan.csv')
    target=pd.read_csv(ROOT/f'data/{SID}_continuum_windows.csv');target['sdss_id']=SID
    target['fieldid']=target.file.str.split('-').str[1].astype(int)
    target['start_tai']=target.start_tai.round().astype('int64')
    times=target.set_index('start_tai');assert times.index.is_unique
    rows=[]
    manifest=json.loads((ROOT/f'data/{SID}_spectral_controls_acquisition.json').read_text())['results']
    assert len(manifest)==len(plan) and all(x['status']=='verified' for x in manifest)
    for row in plan.itertuples():
        with fits.open(ROOT/f'raw/{SID}_spectral_controls'/row.spec_file) as h:
            for k,hd in enumerate(h):
                if not hd.name.startswith('MJD_EXP_'):continue
                start=int(round(hd.header['TAI-BEG']))
                if start not in times.index:continue
                t=times.loc[start]
                r=dict(sdss_id=int(row.sdss_id),file=row.spec_file,hdu=k,mjd=row.mjd,fieldid=row.fieldid,start_tai=start,
                    time=t.time,exptime=float(hd.header['EXPTIME']),split=t.split)
                for name,interval in window.WINDOWS.items():
                    for key,value in window.measure(hd.data,interval).items():r[name+'_'+key]=value
                rows.append(r)
    controls=pd.DataFrame(rows)
    for red in ['red','infrared']:
        obs='blue_'+red;controls[obs]=controls.blue_mag-controls[red+'_mag'];controls['e_'+obs]=np.hypot(controls.blue_error,controls[red+'_error'])
    controls.to_csv(ROOT/f'data/{SID}_continuum_control_measurements.csv',index=False)
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];tests=[];comparison=[];nulls=[]
    for red in ['red','infrared']:
        obs='blue_'+red
        for coverage in [.95,.98,.9]:
            c=controls[(controls.blue_coverage>=coverage)&(controls[red+'_coverage']>=coverage)].dropna(subset=[obs,'e_'+obs]).copy()
            c['error']=np.hypot(c['e_'+obs],.01);c['value']=c[obs];c['demeaned']=np.nan
            for (sid,file),g in c.groupby(['sdss_id','file']):c.loc[g.index,'demeaned']=g.value-np.average(g.value,weights=1/g.error**2)
            common=[]
            for start,g in c.groupby('start_tai'):
                if len(g)<2:continue
                w=1/g.error**2;mean=np.average(g.demeaned,weights=w);scatter=max(1,np.sum(w*(g.demeaned-mean)**2)/(len(g)-1))
                common.append(dict(start_tai=int(start),common_mode=float(mean),common_error=float(np.sqrt(scatter/w.sum())),n_controls=len(g)))
            part=target[(target.blue_coverage>=coverage)&(target[red+'_coverage']>=coverage)].dropna(subset=[obs,'e_'+obs]).merge(pd.DataFrame(common),on='start_tai',validate='one_to_one')
            part['error']=np.sqrt(part['e_'+obs]**2+part.common_error**2+.02**2);part['value']=part[obs]
            part.to_csv(ROOT/f'data/{SID}_continuum_common_{obs}_{coverage:.2f}.csv',index=False)
            if coverage==.95:
                for sid,g in c.groupby('sdss_id'):
                    try:
                        a=fit(g,frozen['selected'][0]['frequency'],tref,'none');a.update(sdss_id=int(sid),observable=obs);comparison.append(a)
                    except ValueError as exc:comparison.append(dict(sdss_id=int(sid),observable=obs,status='unidentifiable',reason=str(exc)))
            train=part[part.split=='discovery_spectra'];valid=part[part.split=='validation_spectra']
            for mode in ['none','subtract','fit']:
                for rank,candidate in enumerate(frozen['selected'],1):
                    f=candidate['frequency'];out=dict(observable=obs,minimum_coverage=coverage,mode=mode,discovery_rank=rank,frequency=f)
                    for name,sample in [('training',train),('validation',valid),('full_exploratory',part)]:
                        try:out[name]=fit(sample,f,tref,mode)
                        except (ValueError,np.linalg.LinAlgError) as exc:out[name]=dict(status='unidentifiable',reason=str(exc),n=len(sample))
                    if 'coefficients' in out['training']:
                        try:out['validation_unchanged_amplitude_phase']=fit(valid,f,tref,mode,out['training']['coefficients'])
                        except (ValueError,np.linalg.LinAlgError) as exc:out['validation_unchanged_amplitude_phase']=dict(status='unidentifiable',reason=str(exc),n=len(valid))
                    tests.append(out)
                # Correlated null uses entire independent nights, keeps all
                # frequencies fixed, and projects fitted nuisances each draw.
                if coverage in [.95,.9] and red=='red':
                    y,X,Q,scale,rank=matrices(part,frozen['selected'][0]['frequency'],tref,mode)
                    bases=[]
                    for candidate in frozen['selected']:
                        _,X,_,_,_=matrices(part,candidate['frequency'],tref,mode)
                        bases.append(np.linalg.svd(X,full_matrices=False)[0][:,:2])
                    basis=np.hstack(bases);_,groups=np.unique(part.mjd,return_inverse=True)
                    rng=np.random.default_rng(20261002);maximum=[];observed=fit(part,frozen['selected'][0]['frequency'],tref,mode)['delta_chi2']
                    for start in range(0,100000,1000):
                        signs=rng.choice([-1.,1.],size=(int(groups.max()+1),1000));z=y[:,None]*signs[groups];z-=Q@(Q.T@z)
                        # Re-estimate conservative null variance after the
                        # sign operation when a common-mode column is fitted.
                        scale_ratio=np.maximum(1.,np.sum(z*z,axis=0)*scale/max(1,len(y)-rank))/scale
                        gain=np.sum((basis.T@z).reshape(20,2,1000)**2,axis=1)/scale_ratio
                        maximum.extend(gain.max(axis=0).tolist())
                    exceed=int(np.sum(np.asarray(maximum)>=observed));nulls.append(dict(observable=obs,minimum_coverage=coverage,mode=mode,draws=100000,n_nights=int(groups.max()+1),exceedances=exceed,observed=observed,plus_one_tail=(exceed+1)/100001,
                        method='Night-sign residual randomization with nuisance projection, variance refit, maximum over20frozen frequencies',limitations='Assumes symmetric independent nights; comparison calibration is held fixed; a diagnostic, not an all-systematics probability.'))
    out=dict(sdss_id=SID,tests=tests,comparison_star_tests=comparison,night_nulls=nulls,
        limitations=['Controls selected using metadata only. No controls or target points removed by measured variability.',
            'Minimum2controls per exposure; errors include their uncertainty and between-control scatter.',
            'Common-mode subtraction assumes shared calibration; free fit permits a separate coefficient in each field.',
            'The two colours and mask variants share photons. Full-sample tests are exploratory; chronological prediction is preserved.'])
    (ROOT/f'data/{SID}_continuum_controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print('PRIMARY',[(r['mode'],r['minimum_coverage'],r.get('full_exploratory',{}).get('delta_chi2'),r.get('validation_unchanged_amplitude_phase',{}).get('delta_chi2')) for r in tests if r['discovery_rank']==1 and r['observable']=='blue_red'],flush=True)
    print('NULLS',nulls,flush=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    d=pd.read_csv(ROOT/f'data/{SID}_continuum_common_blue_red_0.95.csv');freq=frozen['selected'][0]['frequency']
    for ax,mode in zip(axs,['none','subtract']):
        for label,g in d.groupby('split'):
            try:y,X,Q,scale,rank=matrices(g,freq,tref,mode)
            except ValueError:continue
            ax.errorbar(((g.time-tref)*freq)%1,y*g.error*np.sqrt(scale),g.error*np.sqrt(scale),fmt='.',label=label,alpha=.7)
        ax.set(xlabel='Phase of frozen ZTF period',ylabel='Blue−red continuum colour residual (mag)',title='Raw target' if mode=='none' else 'Same-exposure calibration subtracted');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_continuum_controls.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
