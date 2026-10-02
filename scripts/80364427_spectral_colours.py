"""Test independently observed SDSS colours at photometry-frozen frequencies.

Primary observable g-r is set by the opposite ZTF phases. First16 of24
chronological nights fit its amplitude/phase; last8 nights are held out.
All25 visits and96 constituent exposures are retained before quality cuts.
Visit offsets remove static calibration differences. No spectral period scan.
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
from speclite.filters import load_filters
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from helium_line_strengths import REJECT_OR
ROOT=Path(__file__).resolve().parents[1];SID=80364427

def fit(d,f,tref,observable='g_r',fixed=None,floor=.01):
    t=d.time.to_numpy();y=d[observable].to_numpy();e=np.hypot(d['e_'+observable].to_numpy(),floor)
    group=d.file.to_numpy();_,ix=np.unique(group,return_inverse=True);w=1/e**2;ws=np.bincount(ix,w)
    def project(v):return v-(np.bincount(ix,w*v)/ws)[ix]
    yy=project(y);scale=max(1.,float(np.sum(yy**2*w)/max(1,len(y)-len(ws))));w=w/scale
    phi=2*np.pi*f*(t-tref);att=np.sinc(f*d.exptime.to_numpy()/86400);X=np.c_[project(att*np.sin(phi)),project(att*np.cos(phi))]
    gram=X.T@(w[:,None]*X);cov=np.linalg.inv(gram);b=cov@(X.T@(w*yy)) if fixed is None else np.array(fixed)
    null=float(np.sum(w*yy**2));stat=float(np.sum(w*(yy-X@b)**2));amp=float(np.hypot(*b));grad=b/max(amp,1e-30)
    return dict(n=len(d),n_visits=len(ws),n_nights=int(d.mjd.nunique()),frequency=f,observable=observable,delta_chi2=null-stat,
        chi2=stat,null_chi2=null,variance_inflation=scale,coefficients=b.tolist(),covariance=cov.tolist(),amplitude_mag=amp,
        amplitude_error=float(np.sqrt(grad@cov@grad)),phase_radians=float(np.arctan2(b[1],b[0])),
        nominal_20_frequency_tail=float(min(1.,20*chi2.sf(max(0,null-stat),2))) if fixed is None else None),yy

def main():
    target=pd.read_csv(ROOT/'data/fast_rotation_extension_target_plan.csv').set_index('sdss_id').loc[SID]
    coord=SkyCoord(target.ra*u.deg,target.dec*u.deg);apo=EarthLocation.from_geodetic(-105.820278*u.deg,32.780278*u.deg,2788*u.m)
    filters=load_filters(*['sdss2010-'+band for band in 'gri']);rows=[];metadata=[]
    for file in sorted((ROOT/f'raw/fast_rotation/{SID}_spectra').glob('*.fits')):
        with fits.open(file) as h:
            c=h[1].data;cw=10**c['loglam'].astype(float);good=(c['ivar']>0)&np.isfinite(c['flux']);template=np.interp(cw,cw[good],c['flux'][good]);count=0
            for k,x in enumerate(h):
                if not x.name.startswith('MJD_EXP_'):continue
                hd=x.header;d=x.data;w=10**d['loglam'].astype(float);flux=d['flux'].astype(float);iv=d['ivar'].astype(float)
                m=(iv>0)&np.isfinite(flux)&(d['and_mask']==0)&((d['or_mask']&REJECT_OR)==0);delta=np.gradient(w)
                tm=Time((hd['TAI-BEG']+hd['TAI-END'])/172800,format='mjd',scale='tai',location=apo)
                row=dict(file=file.name,hdu=k,mjd=int(h[0].header['MJD']),fieldid=int(h[0].header.get('FIELDID',file.name.split('-')[1])),
                         start_tai=float(hd['TAI-BEG']),time=float((tm.tdb+tm.light_travel_time(coord)).jd-2458000),exptime=float(hd['EXPTIME']))
                for band,response in zip('gri',filters):
                    weight=np.interp(w,response.wavelength,response.response,left=0,right=0)*w*delta
                    template_on_grid=np.interp(w,cw,template)
                    denom=np.sum(weight*template_on_grid);num=np.sum(weight[m]*template_on_grid[m]);coverage=num/denom
                    total=np.sum(weight[m]*flux[m]);var=np.sum(weight[m]**2/iv[m]);corrected=total*denom/num
                    row['coverage_'+band]=coverage
                    if min(total,corrected,num,denom)>0 and coverage>=.98:
                        row[band]=-2.5*np.log10(corrected);row['e_'+band]=2.5/np.log(10)*np.sqrt(var)/total
                    else:row[band]=np.nan;row['e_'+band]=np.nan
                rows.append(row);count+=1
            metadata.append(dict(file=file.name,mjd=int(h[0].header['MJD']),n_exposures=count))
    d=pd.DataFrame(rows);assert len(metadata)==25 and d.mjd.nunique()==24
    d['g_r']=d.g-d.r;d['e_g_r']=np.hypot(d.e_g,d.e_r);d['g_i']=d.g-d.i;d['e_g_i']=np.hypot(d.e_g,d.e_i)
    nights=sorted(d.mjd.unique());split=nights[16];d['split']=np.where(d.mjd<split,'discovery_spectra','validation_spectra')
    d.to_csv(ROOT/f'data/{SID}_spectral_colours.csv',index=False)
    usable=d.dropna(subset=['g_r','e_g_r'])
    split_counts=usable.groupby('split').size().to_dict()
    if min(split_counts.get('discovery_spectra',0),split_counts.get('validation_spectra',0))<5:
        result=dict(sdss_id=SID,status='insufficient_usable_passband_coverage',n_input=len(d),
            n_primary_by_split=split_counts,reason='Primary g-r is not identifiable after visit offsets in the validation sample.',
            continuation='Any continuum-window experiment is a separate measurement, not a successful result of this passband test.')
        (ROOT/f'data/{SID}_spectral_colour_tests.json').write_text(json.dumps(result,indent=2)+'\n')
        print(result,flush=True);return
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];tests=[]
    for observable in ['g_r','g_i']:
        part=d[np.isfinite(d[observable])&np.isfinite(d['e_'+observable])].copy();train=part[part.split=='discovery_spectra'];valid=part[part.split=='validation_spectra']
        for rank,row in enumerate(frozen['selected'],1):
            f=row['frequency'];a,_=fit(train,f,tref,observable);v,_=fit(valid,f,tref,observable);prediction,_=fit(valid,f,tref,observable,a['coefficients']);full,_=fit(part,f,tref,observable)
            tests.append(dict(discovery_rank=rank,frequency=f,observable=observable,training=a,validation=v,validation_unchanged_amplitude_phase=prediction,full_exploratory=full))
    result=dict(sdss_id=SID,n_input=len(d),n_visits=len(metadata),n_nights=len(nights),first_validation_mjd=int(split),metadata=metadata,tests=tests,
        measurement='Photon-weighted SDSS2010 passbands;constant zero points omitted; fluxes corrected for at most2percent missing throughput using phase-averaged visit coadd.',
        phase_convention='coefficients=[sin,cos],phase=atan2(cos_coefficient,sin_coefficient),same convention asZTF.',
        limitations=['Colours cancel grey throughput but not wavelength-dependent calibration.',
        'Coadds supply only the small missing-pixel correction; their photons are not additional measurements.',
        'Per-visit offsets and0.01mag colour floor; variance inflation under null independently in each data split.',
        'g-r is primary;g-i is a correlated sensitivity check.20 frozen frequencies, no spectrum-driven frequency refinement.',
        'The chronological split is defined before these colour measurements, but spectral overview plots were already inspected.'])
    (ROOT/f'data/{SID}_spectral_colour_tests.json').write_text(json.dumps(result,indent=2)+'\n')
    print('inputs',len(d),'usable',d[['g_r','g_i']].notna().sum().to_dict(),'validation starts',split,flush=True)
    print('Primary',json.dumps([r for r in tests if r['discovery_rank']==1],indent=2),flush=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4));f=frozen['selected'][0]['frequency']
    for ax,observable in zip(axs,['g_r','g_i']):
        for label,part in d.dropna(subset=[observable]).groupby('split'):
            _,yy=fit(part,f,tref,observable);err=np.hypot(part['e_'+observable],.01);ax.errorbar(((part.time-tref)*f)%1,yy,err,fmt='.',alpha=.65,label=label)
        ax.set(xlabel='Phase of frozen ZTF period',ylabel=observable.replace('_','−')+' minus visit mean (mag)',title='Independent spectral colour');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_spectral_colours.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()
