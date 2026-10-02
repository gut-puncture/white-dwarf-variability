"""Same-exposure control audit of the training-selected4100A shape feature."""
from pathlib import Path
import os,json,importlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
scan=importlib.import_module('80364427_spectral_feature_scan');reg=scan.regression;ROOT=scan.ROOT;SID=80364427

def measure_controls():
    plan=pd.read_csv(ROOT/f'data/{SID}_spectral_control_plan.csv');times=pd.read_csv(ROOT/f'data/{SID}_continuum_windows.csv')
    times['start_tai']=times.start_tai.round().astype('int64');times=times.set_index('start_tai');rows=[]
    for row in plan.itertuples():
        with fits.open(ROOT/f'raw/{SID}_spectral_controls'/row.spec_file) as h:
            for k,hd in enumerate(h):
                if not hd.name.startswith('MJD_EXP_'):continue
                start=int(round(hd.header['TAI-BEG']))
                if start not in times.index:continue
                time=times.loc[start]
                for feature in ['4100','4400','4700']:
                    r=scan.measure(hd.data,scan.WINDOWS[feature])
                    if r:rows.append(dict(sdss_id=int(row.sdss_id),file=row.spec_file,hdu=k,mjd=row.mjd,fieldid=row.fieldid,start_tai=start,time=time.time,exptime=hd.header['EXPTIME'],split=time.split,feature=feature,**r))
    d=pd.DataFrame(rows);d.to_csv(ROOT/f'data/{SID}_feature_control_measurements.csv',index=False);return d

def fit(d,f,tref,mode,coefficients=None):
    r=reg.fit(d,f,tref,mode,coefficients);r['amplitude_angstrom']=r.pop('amplitude_mag');r['amplitude_error_angstrom']=r.pop('amplitude_error');return r

def main():
    controls=measure_controls();target=pd.concat([pd.read_csv(ROOT/f'data/{SID}_spectral_features_{part}.csv',dtype={'feature':str}) for part in ['training','validation']]);target['fieldid']=target.file.str.split('-').str[1].astype(int);target['start_tai']=target.start_tai.round().astype('int64')
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];frequency=frozen['selected'][0]['frequency'];tests=[];comparisons=[];nulls=[]
    for feature in ['4100','4400','4700']:
        c=controls[controls.feature==feature].copy();c['value']=c.ew;c['error']=np.hypot(c.error,.5);c['demeaned']=np.nan
        for (sid,file),g in c.groupby(['sdss_id','file']):c.loc[g.index,'demeaned']=g.value-np.average(g.value,weights=1/g.error**2)
        for sid,g in c.groupby('sdss_id'):
            try:r=fit(g,frequency,tref,'none');r.update(sdss_id=int(sid),feature=feature);comparisons.append(r)
            except (ValueError,np.linalg.LinAlgError):comparisons.append(dict(sdss_id=int(sid),feature=feature,status='unidentifiable'))
        common=[]
        for start,g in c.groupby('start_tai'):
            if len(g)<2:continue
            w=1/g.error**2;mean=np.average(g.demeaned,weights=w);scale=max(1,np.sum(w*(g.demeaned-mean)**2)/(len(g)-1))
            common.append(dict(start_tai=int(start),common_mode=float(mean),common_error=float(np.sqrt(scale/w.sum())),n_controls=len(g)))
        d=target[target.feature==feature].merge(pd.DataFrame(common),on='start_tai',validate='one_to_one');d['value']=d.ew;d['error']=np.sqrt(d.error**2+d.common_error**2+.5**2)
        d.to_csv(ROOT/f'data/{SID}_feature_common_{feature}.csv',index=False)
        train=d[d.split=='discovery_spectra'];valid=d[d.split=='validation_spectra']
        for mode in ['none','subtract','fit']:
            a=fit(train,frequency,tref,mode);v=fit(valid,frequency,tref,mode);pred=fit(valid,frequency,tref,mode,a['coefficients']);full=fit(d,frequency,tref,mode)
            tests.append(dict(feature=feature,role='training_selected' if feature=='4100' else 'full_data_selected_exploratory',mode=mode,training=a,validation=v,validation_unchanged_amplitude_phase=pred,full_exploratory=full))
            if feature=='4100':
                y,X,Q,scale,rank=reg.matrices(d,frequency,tref,mode);basis=np.linalg.svd(X,full_matrices=False)[0][:,:2];_,night=np.unique(d.mjd,return_inverse=True);rng=np.random.default_rng(20261002);exceed=0;maximum=-np.inf
                for start in range(0,100000,1000):
                    z=y[:,None]*rng.choice([-1.,1.],size=(int(night.max()+1),1000))[night];z-=Q@(Q.T@z)
                    scale_ratio=np.maximum(1,np.sum(z*z,axis=0)*scale/(len(y)-rank))/scale;gain=np.sum((basis.T@z)**2,axis=0)/scale_ratio
                    exceed+=int(sum(gain>=full['delta_chi2']));maximum=max(maximum,float(max(gain)))
                nulls.append(dict(feature=feature,mode=mode,draws=100000,n_nights=int(night.max()+1),observed=full['delta_chi2'],exceedances=exceed,plus_one_tail=(exceed+1)/100001,max_simulated=maximum,
                    interpretation='Exploratory full-sample fixed-feature/frequency night-sign null; does not rerun620trial training selection, so it is not a discovery FAP.'))
    out=dict(tests=tests,comparison_star_tests=comparisons,full_sample_diagnostic_nulls=nulls,
        limitations=['Primary4100Afeature/frequency chosen in training spectra;4400/4700selected after full-sample inspection and remain exploratory.',
            'Common-mode subtraction and fitting reuse the same photons; ensemble scatter and estimated mean errors are propagated approximately.',
            'Phenomenological EW includes local spectral curvature; feature labels are wavelengths, not atomic line identifications.',
            '0.5Aerrorfloor and null variance inflation, no clipping by flux or phase.'])
    (ROOT/f'data/{SID}_feature_controls.json').write_text(json.dumps(out,indent=2)+'\n')
    print('TARGET',[(r['feature'],r['mode'],r['full_exploratory']['n'],r['full_exploratory']['delta_chi2'],r['validation_unchanged_amplitude_phase']['delta_chi2']) for r in tests],flush=True)
    print('CONTROLS4100',[(r['sdss_id'],r.get('delta_chi2'),r.get('amplitude_angstrom')) for r in comparisons if r['feature']=='4100'],flush=True);print('NULLS',nulls,flush=True)
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,feature in zip(axs,['4100','4400','4700']):
        d=pd.read_csv(ROOT/f'data/{SID}_feature_common_{feature}.csv')
        for label,g in d.groupby('split'):
            y,X,Q,scale,rank=reg.matrices(g,frequency,tref,'subtract');ax.errorbar(((g.time-tref)*frequency)%1,y*g.error*np.sqrt(scale),g.error*np.sqrt(scale),fmt='.',alpha=.65,label=label)
        ax.set(title=feature+'Å'+(' selected in training' if feature=='4100' else ' exploratory'),xlabel='Phase of frozen ZTF period',ylabel='Feature EW minus visit mean (Å)')
    axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_feature_controls.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
