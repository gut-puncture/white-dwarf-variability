"""Independent, fixed-period feature-strength tests in SDSS exposure spectra.

Feature and sideband intervals are fixed from the coadded spectrum, before
inspecting exposure strengths or phase. Feature names do not assert atomic
identifications in this strongly magnetic atmosphere. Four primary features,
20 photometry-frozen frequencies; night offsets are always included.
"""
from pathlib import Path
import os,json
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
ROOT=Path(__file__).resolve().parents[1];SID=57829633
# (line interval, blue continuum, red continuum), vacuum Angstrom.
FEATURES={
 'blue4200':((4160,4240),(4100,4140),(4270,4310)),
 'blue4800':((4780,4830),(4730,4760),(4850,4880)),
 'blue4960':((4915,4990),(4865,4900),(5030,5065)),
 'red6600':((6480,6660),(6350,6440),(6690,6780)),
}
CONTROLS={'continuum5500':((5470,5530),(5400,5450),(5560,5610)),
          'continuum6150':((6120,6180),(6050,6100),(6200,6250))}

def measure(d,intervals):
    (a,b),(l0,l1),(r0,r1)=intervals;w=10**d['loglam'].astype(float);y=d['flux'].astype(float);iv=d['ivar'].astype(float)
    good=(iv>0)&np.isfinite(y)&(d['and_mask']==0)&((d['or_mask']&REJECT_OR)==0)
    line=(w>=a)&(w<b);side=((w>=l0)&(w<l1))|((w>=r0)&(w<r1))
    if np.sum(good&line)<.9*sum(line) or np.sum(good&side)<.8*sum(side):return None
    line&=good;side&=good;cen=(a+b)/2;span=r1-l0;x=(w-cen)/span
    X=np.c_[np.ones(sum(side)),x[side]];cov=np.linalg.inv(X.T@(iv[side,None]*X));coef=cov@(X.T@(iv[side]*y[side]))
    XL=np.c_[np.ones(sum(line)),x[line]];c=XL@coef
    if np.any(c<=0):return None
    dw=np.gradient(w)[line];ew=np.sum(dw*(1-y[line]/c));grad=np.sum((dw*y[line]/c**2)[:,None]*XL,axis=0)
    var=np.sum(dw**2/iv[line]/c**2)+grad@cov@grad
    return dict(ew=float(ew),error=float(np.sqrt(var)),sideband_rchi2=float(np.sum((y[side]-X@coef)**2*iv[side])/(sum(side)-2)),n_pixels=int(sum(line)))

def fit(d,f,tref,trend=False,floor=0):
    t=d.time.to_numpy();y=d.ew.to_numpy();e=np.hypot(d.error.to_numpy(),floor);night=d.mjd.to_numpy();nights=np.unique(night)
    columns=[(night==n).astype(float) for n in nights]
    if trend:
        for n in nights:
            mask=night==n
            if sum(mask)>=3:columns.append(np.where(mask,(t-np.mean(t[mask]))/np.ptp(t[mask]),0.))
    B=np.array(columns).T;theta=2*np.pi*f*(t-tref);atten=np.sinc(f*d.exptime.to_numpy()/86400)
    X=np.c_[B,atten*np.cos(theta),atten*np.sin(theta)];A=X/e[:,None];N=B/e[:,None]
    b=np.linalg.lstsq(A,y/e,rcond=None)[0];n=np.linalg.lstsq(N,y/e,rcond=None)[0]
    chi=np.sum(((y-X@b)/e)**2);null=np.sum(((y-B@n)/e)**2);cov=np.linalg.pinv(A.T@A)[-2:,-2:]
    amp=float(np.hypot(*b[-2:]));grad=b[-2:]/amp
    return dict(frequency=f,delta_chi2=float(null-chi),null_chi2=float(null),chi2=float(chi),n=len(d),night_offsets=len(nights),
        extra_night_trend=trend,error_floor_angstrom=floor,amplitude_angstrom=amp,formal_amplitude_error=float(np.sqrt(grad@cov@grad)),phase_radians=float(np.arctan2(b[-1],b[-2])),
        coefficients=b[-2:].tolist(),nominal_80_tests_tail=float(min(1.,80*chi2.sf(max(0,null-chi),2))))

def main():
    target=pd.read_csv(ROOT/'data/fast_rotation_extension_target_plan.csv').set_index('sdss_id').loc[SID]
    coord=SkyCoord(target.ra*u.deg,target.dec*u.deg);apo=EarthLocation.from_geodetic(-105.820278*u.deg,32.780278*u.deg,2788*u.m)
    rows=[]
    for file in sorted((ROOT/f'raw/fast_rotation/{SID}_spectra').glob('*.fits')):
        with fits.open(file) as h:
            for k,x in enumerate(h):
                if not x.name.startswith('MJD_EXP_'):continue
                hd=x.header;mid=(hd['TAI-BEG']+hd['TAI-END'])/172800
                time=Time(mid,format='mjd',scale='tai',location=apo);t=float((time.tdb+time.light_travel_time(coord)).jd-2458000)
                for name,interval in {**FEATURES,**CONTROLS}.items():
                    r=measure(x.data,interval)
                    if r:rows.append(dict(file=file.name,hdu=k,mjd=int(h[0].header['MJD']),time=t,exptime=float(hd['EXPTIME']),feature=name,role='primary' if name in FEATURES else 'continuum_control',**r))
    d=pd.DataFrame(rows);d.to_csv(ROOT/f'data/{SID}_spectral_feature_strengths.csv',index=False)
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];tests=[]
    for name,g in d.groupby('feature'):
        for rank,row in enumerate(frozen['selected'],1):
            for label,trend,floor in [('primary',False,0),('night_slope',True,0),('one_angstrom_floor',False,1.)]:
                r=fit(g,row['frequency'],tref,trend,floor);r.update(feature=name,role=g.role.iloc[0],variant=label,discovery_rank=rank);tests.append(r)
    output=dict(sdss_id=SID,features=FEATURES,continuum_controls=CONTROLS,n_spectra=int(d[['file','hdu']].drop_duplicates().shape[0]),tests=tests,
        limitations=['Local linear continuum model and extraction errors may cause apparent feature changes.',
        'Sideband-fit uncertainty propagated; wavelength resampling covariance is not supplied by the archive.',
        '20 frozen ZTF frequencies and4 primary features count as80 tests; controls and nuisance variants are diagnostics.',
        'No frequency refinement or phase-selected removal of exposures. Same-photon coadds excluded.'])
    (ROOT/f'data/{SID}_spectral_modulation.json').write_text(json.dumps(output,indent=2)+'\n')
    print('strengths',d.groupby('feature').agg(n=('ew','size'),mean=('ew','mean'),scatter=('ew','std'),median_error=('error','median')).to_string(),flush=True)
    print('primary frequency',[(x['feature'],x['variant'],round(x['delta_chi2'],3),x['nominal_80_tests_tail']) for x in tests if x['discovery_rank']==1],flush=True)
    print('best corrected primary',sorted([x for x in tests if x['role']=='primary' and x['variant']=='primary'],key=lambda x:x['delta_chi2'],reverse=True)[:3],flush=True)
    fig,axs=plt.subplots(2,3,figsize=(13,7));freq=frozen['selected'][0]['frequency']
    for ax,(name,g) in zip(axs.flat,d.groupby('feature')):
        for night,s in g.groupby('mjd'):
            wt=1/s.error**2;v=s.ew-np.average(s.ew,weights=wt)
            ax.errorbar(((s.time-tref)*freq)%1,v,s.error,fmt='.',label=str(night))
        ax.set(title=name,xlabel='Phase of frozen primary ZTF period',ylabel='Feature EW minus nightly mean (Å)');ax.axhline(0,color='gray',lw=.5)
    axs[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_spectral_modulation.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
