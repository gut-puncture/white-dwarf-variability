"""Discovery/validation search for the frozen compact-star pilot.

Discovery frequencies never use the later25%of observing nights. Data and each
stage are checkpointed independently. Reported probabilities are screening
diagnostics and do not establish a spin period, source attribution or novelty.
"""
from pathlib import Path
import os,json,hashlib,time,argparse,importlib,traceback
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar,brentq
from scipy.signal import find_peaks
from scipy.stats import chi2
from astropy.coordinates import SkyCoord,EarthLocation
from astropy.time import Time
from astropy import units as u
from astropy.timeseries import LombScargle
from astropy.utils import iers
iers.conf.auto_download=False
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'data/fast_rotation';DEST.mkdir(exist_ok=True)
audit=importlib.import_module('74835110_photometric_audit')
VERSION=2

def prepare(row):
    sid=int(row['sdss_id']);rawpath=ROOT/f'raw/fast_rotation/{sid}_ztf.csv'
    raw=pd.read_csv(rawpath,dtype={'oid':str});raw_sha=hashlib.sha256(rawpath.read_bytes()).hexdigest()
    if not len(raw):return None,dict(sdss_id=sid,status='no ZTF observations',input_sha256=raw_sha)
    objects=raw.groupby('oid').agg(ra=('ra','median'),dec=('dec','median'),mjd=('mjd','median'),n=('mjd','size')).reset_index()
    # An OID may refer to a fixed reference-image position. Match its centroid
    # to the whole Gaia motion path, not every date to a stationary1.5arcsec cone.
    x=(objects.ra-float(row['ra']))*np.cos(np.deg2rad(row['dec']))*3600;y=(objects.dec-float(row['dec']))*3600
    vx,vy=float(row['pmra'])/1000,float(row['pmdec'])/1000
    lo=(raw.mjd.min()-57388)/365.25;hi=(raw.mjd.max()-57388)/365.25
    dt=np.clip((x*vx+y*vy)/max(vx*vx+vy*vy,1e-12),lo,hi)
    objects['distance_to_motion_path_arcsec']=np.hypot(x-vx*dt,y-vy*dt);objects['associated']=objects.distance_to_motion_path_arcsec.lt(1.5)
    objects.to_csv(DEST/f'{sid}_object_associations.csv',index=False)
    d=raw[raw.oid.isin(objects.loc[objects.associated,'oid'])].copy()
    valid=(d.catflags==0)&d.magerr.gt(0)&d.magerr.lt(.15)&np.isfinite(d.mag)&np.isfinite(d.mjd)&d.exptime.gt(0)
    d=d[valid&d.filtercode.isin(['zg','zr'])].sort_values('magerr').drop_duplicates(['expid','filtercode']).sort_values('mjd').reset_index(drop=True)
    if len(d)<170:return None,dict(sdss_id=sid,status='too few usable observations',n=len(d),n_raw=len(raw),input_sha256=raw_sha)
    tm=Time(d.mjd.to_numpy()+d.exptime.to_numpy()/172800,format='mjd',scale='utc',location=EarthLocation.from_geodetic(-116.863*u.deg,33.356*u.deg,1706*u.m))
    coord=SkyCoord(float(row['ra'])*u.deg,float(row['dec'])*u.deg)
    d['time']=(tm.tdb+tm.light_travel_time(coord)).jd-2458000
    hjd=tm.utc.jd+tm.light_travel_time(coord,kind='heliocentric').to_value(u.day)
    timing=(hjd-d.hjd.to_numpy())*86400
    nights=np.unique(np.floor(d.mjd));cut=nights[int(np.floor(.75*len(nights)))];d['split']=np.where(np.floor(d.mjd)<cut,'discovery','validation')
    counts=d.groupby(['filtercode','split']).size().unstack(fill_value=0)
    bands=counts.index[(counts.discovery>=50)&(counts.validation>=15)].tolist();d=d[d.filtercode.isin(bands)].copy().reset_index(drop=True)
    ntrain=int(d.split.eq('discovery').sum());nval=int(d.split.eq('validation').sum())
    meta=dict(sdss_id=sid,gaia_dr3_source_id=str(row['gaia_dr3_source_id']),input_sha256=raw_sha,method_version=VERSION,n_raw=len(raw),n_clean=len(d),n_discovery=ntrain,n_validation=nval,
        bands=bands,split_night_mjd=float(cut),n_nights=len(nights),role=row['role'],known_control=row['known_control'],previously_examined_ztf=bool(row['previously_examined_ztf']),
        reconstructed_hjd_minus_archive_seconds_percentiles=np.percentile(timing,[0,50,100]).tolist(),exposure_seconds=sorted(d.exptime.unique().tolist()),
        association='Within1.5arcsec of Gaia proper-motion track; OID centroids retained, no variability-based association.',
        time_system='Exposure-midpoint BJD TDB−2458000; MJD_UTC start plus half exposure; Palomar geodetic position.')
    d.to_csv(DEST/f'{sid}_prepared.csv.gz',index=False);(DEST/f'{sid}_preparation.json').write_text(json.dumps(meta,indent=2)+'\n')
    if ntrain<120 or nval<50 or not bands:meta['status']='insufficient discovery/validation coverage';return None,meta
    return d,meta

class SplitModel:
    def __init__(self,d,tref,systematics=True,jitter=True):
        self.d=d.reset_index(drop=True);self.t=self.d.time.to_numpy()-tref;self.tref=tref;self.bands=sorted(self.d.filtercode.unique())
        self.y=self.d.mag.to_numpy();self.error=np.hypot(self.d.magerr.to_numpy(),.005)
        self.x0,self.names=audit.nuisance(self.d,systematics);self.extra={}
        beta=np.linalg.lstsq(self.x0/self.error[:,None],self.y/self.error,rcond=None)[0];res=self.y-self.x0@beta
        for band in self.bands:
            m=self.d.filtercode.eq(band).to_numpy();dof=max(1,int(m.sum())-np.linalg.matrix_rank(self.x0[m]))
            fn=lambda j:np.sum(res[m]**2/(self.error[m]**2+j*j))-dof
            val=brentq(fn,0,2) if jitter and fn(0)>0 else 0.;self.extra[band]=float(val);self.error[m]=np.hypot(self.error[m],val)
        wx=self.x0/self.error[:,None];u,s,v=np.linalg.svd(wx,full_matrices=False);rank=np.sum(s>s[0]*np.finfo(float).eps*max(wx.shape));self.q=u[:,:rank]
        self.ry=self.project(self.y/self.error);self.null_chi2=float(self.ry@self.ry);self.rank=int(rank);self.residual=self.ry*self.error
        self.mask={b:self.d.filtercode.eq(b).to_numpy() for b in self.bands}

    def project(self,x):return x-self.q@(self.q.T@x)

    def design(self,f):
        phase=2*np.pi*f*self.t;response=np.sinc(f*self.d.exptime.to_numpy()/86400)
        return np.array([value for b in self.bands for value in [self.mask[b]*np.sin(phase)*response/self.error,self.mask[b]*np.cos(phase)*response/self.error]]).T

    def fit(self,f,details=False):
        z=self.project(self.design(f));beta=np.linalg.lstsq(z,self.ry,rcond=None)[0];res=self.ry-z@beta
        value=float(res@res)
        if not details:return value
        inverse=np.linalg.pinv(z);cov=inverse@inverse.T;params=[]
        for i,b in enumerate(self.bands):
            ab=beta[2*i:2*i+2];ca=cov[2*i:2*i+2,2*i:2*i+2];amp=np.hypot(*ab)
            grad=np.array([-ab[1],ab[0]])/max(amp**2,1e-30)
            params.append(dict(band=b,amplitude_mag=float(amp),phase_radians=float(np.arctan2(ab[1],ab[0])),phase_error_radians=float(np.sqrt(max(0,grad@ca@grad)))))
        return dict(frequency=float(f),period_seconds=float(86400/f),chi2=value,null_chi2=self.null_chi2,delta_chi2=self.null_chi2-value,
            n=len(self.d),nuisance_rank=self.rank,parameters=params,coefficients=beta.tolist(),covariance=cov.tolist(),jitter_mag=self.extra,
            nominal_fixed_frequency_tail=float(chi2.sf(self.null_chi2-value,2*len(self.bands))))

    def predict_gain(self,f,beta):
        pred=self.project(self.design(f))@np.array(beta)
        return float(self.null_chi2-np.sum((self.ry-pred)**2))

def discovery_scan(model,meta,fmax):
    sid=meta['sdss_id'];path=DEST/f'{sid}_discovery_scan.json';step=1/(np.ptp(model.t)*5);n=int(np.floor((fmax-.01)/step))+1;chunk=500000
    if path.exists():
        result=json.loads(path.read_text());assert result['input_sha256']==meta['input_sha256'] and result['fmax']==fmax and result['method_version']==VERSION
    else:
        result=dict(method_version=VERSION,input_sha256=meta['input_sha256'],fmax=fmax,step=step,n_frequencies=n,chunks=[],tref=model.tref,n_discovery=len(model.d))
    for k,start in enumerate(range(0,n,chunk)):
        if k<len(result['chunks']):continue
        indices=np.arange(start,min(n,start+chunk));freq=.01+step*indices;total=np.zeros(len(freq));band_peaks=[]
        for b,m in model.mask.items():
            y=model.residual[m];err=model.error[m];w=1/err**2;y=y-np.average(y,weights=w);q0=np.sum(w*y*y)
            power=LombScargle(model.t[m],y,err).power(freq,method='fast',assume_regular_frequency=True);score=q0*power;total+=score
            ix=find_peaks(score)[0];ix=ix[np.argsort(score[ix])[-8:]]
            band_peaks.extend([dict(frequency=float(freq[i]),score=float(score[i]),selection=b) for i in ix])
        ix=find_peaks(total)[0];ix=np.unique(np.r_[ix[np.argsort(total[ix])[-30:]],np.argmax(total)])
        peaks=[dict(frequency=float(freq[i]),score=float(total[i]),selection='summed') for i in ix]+band_peaks
        result['chunks'].append(dict(index=k,start=start,stop=int(indices[-1])+1,peaks=peaks,maximum=float(total.max())))
        path.write_text(json.dumps(result,indent=2)+'\n');print(sid,'chunk',k+1,'/',int(np.ceil(n/chunk)),'maximum',float(total.max()),flush=True)
    return result

def one(row,fmax,family_size=44):
    sid=int(row['sdss_id']);dest=DEST/f'{sid}_result.json'
    if dest.exists():
        old=json.loads(dest.read_text())
        if old.get('method_version')==VERSION and old.get('fmax')==fmax:
            old['screening_family_size']=family_size
            old['original_search_role']=old.get('original_search_role',old.get('role'))
            old['role']=row['role'];old['known_control']=row['known_control']
            old['previously_examined_ztf']=bool(row['previously_examined_ztf'])
            for check in old.get('checks',[]):
                check['nominal_family_adjusted_validation_tail']=min(1.,family_size*20*check['validation']['nominal_fixed_frequency_tail'])
            return old
    d,meta=prepare(row)
    if d is None:meta.update(method_version=VERSION,fmax=fmax);dest.write_text(json.dumps(meta,indent=2)+'\n');return meta
    start=time.monotonic();train=d[d.split.eq('discovery')];valid=d[d.split.eq('validation')];tref=float(train.time.mean());model=SplitModel(train,tref)
    scan=discovery_scan(model,meta,fmax);step=scan['step'];ranked=[]
    for ch in scan['chunks']:ranked.extend(ch['peaks'])
    candidates=[]
    for r in sorted(ranked,key=lambda r:r['score'],reverse=True):
        if all(abs(r['frequency']-c)>3*step for c in candidates):candidates.append(r['frequency'])
        if len(candidates)>=140:break
    refined=[]
    for f in candidates:
        # Optimize an offset: scipy's bounded solver includes a relative
        # tolerance which is otherwise material at >1000 cycles/day.
        opt=minimize_scalar(lambda offset:model.fit(f+offset),bounds=(max(.001-f,-2*step),min(fmax-f,2*step)),method='bounded',options={'xatol':1e-11})
        fit=model.fit(f+opt.x,True)
        if all(abs(fit['frequency']-r['frequency'])>3*step for r in refined):refined.append(fit)
    refined.sort(key=lambda r:r['chi2']);frozen=refined[:20]
    # This file is written before a validation model or statistic is created.
    (DEST/f'{sid}_frozen_discovery_candidates.json').write_text(json.dumps(dict(metadata=meta,tref=tref,selected=frozen),indent=2)+'\n')
    validation=SplitModel(valid,tref);checks=[]
    for r in frozen:
        v=validation.fit(r['frequency'],True);fixed=validation.predict_gain(r['frequency'],r['coefficients'])
        nominal=min(1.,family_size*20*v['nominal_fixed_frequency_tail'])
        checks.append(dict(frequency=r['frequency'],period_seconds=r['period_seconds'],discovery=r,validation=v,
            validation_gain_with_discovery_amplitude_phase=fixed,nominal_family_adjusted_validation_tail=nominal))
    result=dict(**meta,status='complete',fmax=fmax,screening_family_size=family_size,elapsed_seconds=time.monotonic()-start,checks=checks,
        interpretation='Exploratory candidates require coherent replication, alias/contamination checks and independent physical interpretation.',
        caveats=['Nominal validation tails assume the adopted residual model; they are not all-systematics discovery probabilities.',
                 'Earlier-examined light curves and known controls are labelled; their holdout is not prospectively unexamined.',
                 'Quality regression and constant-model jitter are conservative but may absorb signal.',
                 'The compact colour/luminosity cut is not a measured white-dwarf mass.'])
    dest.write_text(json.dumps(result,indent=2)+'\n')
    best=checks[0];print('COMPLETE',sid,'P',best['period_seconds'],'discovery',best['discovery']['delta_chi2'],'validation',best['validation']['delta_chi2'],'fixed',best['validation_gain_with_discovery_amplitude_phase'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--ids',nargs='*',type=int);p.add_argument('--fmax',type=float,default=1440.);a=p.parse_args()
    plan=pd.read_csv(ROOT/'data/fast_rotation_target_plan.csv',dtype={'gaia_dr3_source_id':str}).fillna({'known_control':''})
    if a.ids:plan=plan[plan.sdss_id.isin(a.ids)]
    results=[]
    for row in plan[plan.eligible].to_dict('records'):
        try:r=one(row,a.fmax)
        except Exception as exc:r=dict(sdss_id=int(row['sdss_id']),status='failed',error=repr(exc),traceback=traceback.format_exc());print(r,flush=True)
        results.append(r)
        tag='_'.join(map(str,a.ids)) if a.ids else 'all'
        (ROOT/f'data/fast_rotation_summary_{tag}.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
