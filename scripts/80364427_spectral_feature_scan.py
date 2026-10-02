"""Training-selected local spectral-shape tests with a chronological holdout."""
from pathlib import Path
import os,json,importlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from astropy.io import fits
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
measure=importlib.import_module('57829633_spectral_modulation').measure
regression=importlib.import_module('80364427_continuum_controls')
ROOT=regression.ROOT;SID=80364427
WINDOWS={str(c):((c-40,c+40),(c-100,c-60),(c+60,c+100)) for c in range(3800,6801,100)}

def collect(metadata):
    rows=[]
    for name,g in metadata.groupby('file'):
        with fits.open(ROOT/f'raw/fast_rotation/{SID}_spectra'/name) as h:
            for r in g.itertuples():
                for name,interval in WINDOWS.items():
                    a=measure(h[r.hdu].data,interval)
                    if a:rows.append(dict(file=r.file,hdu=r.hdu,mjd=r.mjd,time=r.time,exptime=r.exptime,start_tai=r.start_tai,split=r.split,feature=name,**a))
    return pd.DataFrame(rows)

def fit(d,f,tref,coefficients=None):
    q=d.copy();q['value']=q.ew;q['error']=np.hypot(q.error,.5)
    r=regression.fit(q,f,tref,'none',coefficients)
    r['amplitude_angstrom']=r.pop('amplitude_mag');r['formal_amplitude_error_angstrom']=r.pop('amplitude_error')
    r['nominal_fixed_frequency_tail']=float(chi2.sf(max(0,r['delta_chi2']),2)) if coefficients is None else None
    return r

def main():
    meta=pd.read_csv(ROOT/f'data/{SID}_continuum_windows.csv');train=collect(meta[meta.split=='discovery_spectra'])
    train.to_csv(ROOT/f'data/{SID}_spectral_features_training.csv',index=False)
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];training=[]
    for feature,g in train.groupby('feature'):
        for rank,c in enumerate(frozen['selected'],1):
            try:r=fit(g,c['frequency'],tref)
            except (ValueError,np.linalg.LinAlgError):continue
            training.append(dict(feature=feature,discovery_rank=rank,frequency=c['frequency'],training=r))
    training.sort(key=lambda x:x['training']['delta_chi2'],reverse=True)
    selection=dict(selected=training[0],ranked_training=training,n_windows=31,n_frequencies=20,first_validation_mjd=59268,
        method='Feature and frequency selected using only first16spectral nights; no spectral frequency refinement.',
        limitations='Follow-up after inspection of continuum variability; not an independently blind discovery campaign.')
    path=ROOT/f'data/{SID}_spectral_features_frozen_selection.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['selected']['feature']==selection['selected']['feature'] and old['selected']['frequency']==selection['selected']['frequency']
    else:path.write_text(json.dumps(selection,indent=2)+'\n')
    print('FROZEN SPECTRAL SELECTION',selection['selected'],flush=True)
    valid=collect(meta[meta.split=='validation_spectra']);valid.to_csv(ROOT/f'data/{SID}_spectral_features_validation.csv',index=False)
    d=pd.concat([train,valid]);tests=[]
    for candidate in training:
        feature=candidate['feature'];f=candidate['frequency'];v=valid[valid.feature==feature];full=d[d.feature==feature];r=dict(candidate)
        for name,part,coef in [('validation',v,None),('validation_unchanged_amplitude_phase',v,candidate['training']['coefficients']),('full_exploratory',full,None)]:
            try:r[name]=fit(part,f,tref,coef)
            except (ValueError,np.linalg.LinAlgError) as exc:r[name]=dict(status='unidentifiable',reason=str(exc),n=len(part))
        tests.append(r)
    out=dict(sdss_id=SID,selected_validation=tests[0],all_tests=tests,windows=WINDOWS,error_floor_angstrom=.5,
        phase_convention='coefficients=[sin,cos],as ZTF',
        limitations=['620feature/frequency trials used in training. The selected feature has one chronological validation test; inspecting other validation tests requires620test accounting.',
        'Each local continuum is linear; neighboring windows share continuum pixels. No atomic identification is asserted.',
        'Per-visit offsets and variance inflation under constant model; masked extraction and resampling systematics remain possible.',
        'Any surviving feature requires same-exposure comparison stars and whole-night nulls.'])
    (ROOT/f'data/{SID}_spectral_features_validation.json').write_text(json.dumps(out,indent=2)+'\n')
    print('SELECTED VALIDATION',json.dumps(tests[0],indent=2),flush=True)
    primary=[r for r in tests if r['discovery_rank']==1];primary.sort(key=lambda r:r.get('full_exploratory',{}).get('delta_chi2',-1),reverse=True)
    print('EXPLORATORY PRIMARY FREQUENCY',[(x['feature'],x['training']['delta_chi2'],x['validation'].get('delta_chi2'),x['full_exploratory'].get('delta_chi2')) for x in primary[:6]],flush=True)
    chosen=tests[0];feature=chosen['feature'];f=chosen['frequency'];fig,ax=plt.subplots(figsize=(7,4))
    for label,g in d[d.feature==feature].groupby('split'):
        q=g.copy();q['value']=q.ew;q['error']=np.hypot(q.error,.5)
        try:y,X,Q,scale,rank=regression.matrices(q,f,tref,'none')
        except ValueError:continue
        ax.errorbar(((q.time-tref)*f)%1,y*q.error*np.sqrt(scale),q.error*np.sqrt(scale),fmt='.',label=label,alpha=.7)
    ax.set(title=f'Training-selected {feature}Å feature; frequency rank{chosen["discovery_rank"]}',xlabel='Phase of frozen frequency',ylabel='Equivalent width minus visit mean (Å)');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_spectral_selected_feature.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
