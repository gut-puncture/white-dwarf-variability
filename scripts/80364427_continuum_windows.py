"""Separate fixed-window diagnostic after full-passband coverage proved inadequate.

Windows were specified from coadded spectra and mask coverage before measuring
exposure flux variation. Primary colour:4000-4400 minus5950-6250 Angstrom.
Require95percent good pixels.98percent and90percent are labelled sensitivity
checks;7750-8050 is a secondary colour. No period refinement from spectra.
"""
from pathlib import Path
import os,json,importlib
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord,EarthLocation
from astropy import units as u
from astropy.utils import iers
iers.conf.auto_download=False
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helium_line_strengths import REJECT_OR
old=importlib.import_module('80364427_spectral_colours')
ROOT=Path(__file__).resolve().parents[1];SID=80364427
WINDOWS={'blue':(4000,4400),'red':(5950,6250),'infrared':(7750,8050)}

def measure(d,interval):
    a,b=interval;wave=10**d['loglam'].astype(float);flux=d['flux'].astype(float);iv=d['ivar'].astype(float)
    region=(wave>=a)&(wave<b);good=region&(iv>0)&np.isfinite(flux)&(d['and_mask']==0)&((d['or_mask']&REJECT_OR)==0)
    coverage=sum(good)/max(1,sum(region))
    result=dict(coverage=coverage)
    if coverage<.9:return result
    x=(wave[good]-(a+b)/2)/(b-a);X=np.c_[np.ones(sum(good)),x]
    gram=X.T@(iv[good,None]*X)
    if np.linalg.cond(gram)>1e8:return result
    cov=np.linalg.inv(gram);coef=cov@(X.T@(iv[good]*flux[good]));rchi=float(np.sum((flux[good]-X@coef)**2*iv[good])/(sum(good)-2))
    if coef[0]<=0:return result
    err=np.sqrt(cov[0,0]*max(1,rchi))
    result.update(mag=float(-2.5*np.log10(coef[0])),error=float(2.5/np.log(10)*err/coef[0]),rchi2=rchi,flux=float(coef[0]))
    return result

def main():
    target=pd.read_csv(ROOT/'data/fast_rotation_extension_target_plan.csv').set_index('sdss_id').loc[SID]
    coord=SkyCoord(target.ra*u.deg,target.dec*u.deg);apo=EarthLocation.from_geodetic(-105.820278*u.deg,32.780278*u.deg,2788*u.m)
    rows=[]
    for file in sorted((ROOT/f'raw/fast_rotation/{SID}_spectra').glob('*.fits')):
        with fits.open(file) as h:
            for k,hd in enumerate(h):
                if not hd.name.startswith('MJD_EXP_'):continue
                header=hd.header;tm=Time((header['TAI-BEG']+header['TAI-END'])/172800,format='mjd',scale='tai',location=apo)
                row=dict(file=file.name,hdu=k,mjd=int(h[0].header['MJD']),start_tai=float(header['TAI-BEG']),
                    time=float((tm.tdb+tm.light_travel_time(coord)).jd-2458000),exptime=float(header['EXPTIME']))
                for name,window in WINDOWS.items():
                    for key,value in measure(hd.data,window).items():row[name+'_'+key]=value
                rows.append(row)
    d=pd.DataFrame(rows);assert len(d)==96 and d.mjd.nunique()==24
    d['split']=np.where(d.mjd<59268,'discovery_spectra','validation_spectra')
    for red in ['red','infrared']:
        obs='blue_'+red;d[obs]=d.blue_mag-d[red+'_mag'];d['e_'+obs]=np.hypot(d.blue_error,d[red+'_error'])
    d.to_csv(ROOT/f'data/{SID}_continuum_windows.csv',index=False)
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];tests=[]
    for red in ['red','infrared']:
        obs='blue_'+red
        for coverage in [.95,.98,.9]:
            part=d[(d.blue_coverage>=coverage)&(d[red+'_coverage']>=coverage)].dropna(subset=[obs,'e_'+obs]).copy()
            train=part[part.split=='discovery_spectra'];valid=part[part.split=='validation_spectra']
            print(obs,'coverage',coverage,'train',len(train),'validation',len(valid),flush=True)
            for rank,candidate in enumerate(frozen['selected'],1):
                f=candidate['frequency'];record=dict(discovery_rank=rank,frequency=f,observable=obs,minimum_coverage=coverage)
                try:
                    a,_=old.fit(train,f,tref,obs,floor=.02);v,_=old.fit(valid,f,tref,obs,floor=.02)
                    prediction,_=old.fit(valid,f,tref,obs,a['coefficients'],floor=.02);full,_=old.fit(part,f,tref,obs,floor=.02)
                    record.update(training=a,validation=v,validation_unchanged_amplitude_phase=prediction,full_exploratory=full)
                except np.linalg.LinAlgError:record.update(status='unidentifiable')
                tests.append(record)
    output=dict(sdss_id=SID,windows=WINDOWS,first_validation_mjd=59268,primary='blue_red at95percent coverage',n_input=len(d),tests=tests,
        measurement='Locally linear continuum flux at fixed window centre; formal covariance inflated if within-window residual chi2 exceeds degrees of freedom.',
        limitations=['Diagnostic continuum colours, not exact SDSS or ZTF passband magnitudes.',
        'Two colours and20frozen frequencies; 95percent is primary, other thresholds are correlated sensitivity checks.',
        'Per-visit offsets remove grey changes; chromatic calibration still requires comparison-star controls.',
        '0.02mag error floor and variance inflation under constant model separately in each split.',
        'No data clipping by flux, phase, or strength. All masks use the previously applied SPPIXMASK bits.'])
    (ROOT/f'data/{SID}_continuum_window_tests.json').write_text(json.dumps(output,indent=2)+'\n')
    print('PRIMARY FREQUENCY',json.dumps([r for r in tests if r['discovery_rank']==1],indent=2),flush=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4));freq=frozen['selected'][0]['frequency']
    for ax,red in zip(axs,['red','infrared']):
        obs='blue_'+red;part=d[(d.blue_coverage>=.95)&(d[red+'_coverage']>=.95)].dropna(subset=[obs,'e_'+obs])
        for label,g in part.groupby('split'):
            fit,yy=old.fit(g,freq,tref,obs,floor=.02)
            ax.errorbar(((g.time-tref)*freq)%1,yy,np.hypot(g['e_'+obs],.02)*np.sqrt(fit['variance_inflation']),fmt='.',alpha=.65,label=label)
        ax.set(xlabel='Phase of frozen primary ZTF period',ylabel=obs.replace('_','−')+' minus visit mean (mag)',title='Fixed continuum windows;95% coverage');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_continuum_windows.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
