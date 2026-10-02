"""Exploratory helium-line centroids with instrumental-width-aware profiles.

Pressure shifts and forbidden components mean these are NOT calibrated centre
of mass velocities. The purpose is to identify spectra for inspection and
initialize a same-star, shared-profile variability test. No formal discovery
probabilities follow from this screen.
"""
from pathlib import Path
import os,json,argparse,concurrent.futures as cf
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd
from astropy.io import fits
from scipy.special import voigt_profile
from scipy.optimize import least_squares
from helium_cohort_io import read_plan
ROOT=Path(__file__).resolve().parents[1];C=299792.458
# NIST strong-line air wavelengths; strongest permitted component per feature.
AIR={'he_4471':4471.479,'he_4922':4921.931,'he_5876':5875.6148,'he_6678':6678.1517}
WIDTH={'he_4471':55,'he_4922':48,'he_5876':55,'he_6678':55}

def air_to_vac(w):
    s2=(1e4/w)**2
    return w*(1+6.4328e-5+2.94981e-2/(146-s2)+2.5540e-4/(41-s2))
REST={k:float(air_to_vac(v)) for k,v in AIR.items()}

def fit_line(d,name):
    rest=REST[name];width=WIDTH[name];wave=10**d['loglam'].astype(float)
    good=(abs(wave-rest)<width)&np.isfinite(d['flux'])&(d['ivar']>0)&(d['and_mask']==0)
    if good.sum()<25:return None
    w=wave[good];f=d['flux'][good].astype(float);iv=d['ivar'][good].astype(float)
    scale=np.percentile(f,85)
    if scale<=0:return None
    y=f/scale;e=1/np.sqrt(iv)/scale;x=(w-rest)/width
    inst=np.median(d['wdisp'][good]*wave[good]*np.log(10)*1e-4)
    if not np.isfinite(inst) or inst<=0:return None
    def calculate(p,full=False):
        v,g,sig=p;centre=rest*np.sqrt((1+v/C)/(1-v/C));sigma=np.hypot(sig,inst)
        prof=voigt_profile(w-centre,sigma,g)/voigt_profile(0,sigma,g)
        X=np.column_stack([np.ones(len(w)),x,-prof]);b=np.linalg.lstsq(X/e[:,None],y/e,rcond=None)[0]
        if b[2]<0:b[:2]=np.linalg.lstsq(X[:,:2]/e[:,None],y/e,rcond=None)[0];b[2]=0
        model=X@b;res=(model-y)/e
        return (res,model,b) if full else res
    fits=[]
    for v in [0,-900,900]:
        r=least_squares(calculate,[v,8,2],bounds=([-2500,.25,0],[2500,80,35]),x_scale=[200,10,3],max_nfev=150)
        fits.append(r)
    f=min(fits,key=lambda a:a.fun@a.fun);chi=float(f.fun@f.fun);dof=len(w)-6
    res,model,b=calculate(f.x,True);X=np.column_stack([np.ones(len(w)),x]);b0=np.linalg.lstsq(X/e[:,None],y/e,rcond=None)[0]
    chi0=float(np.sum(((y-X@b0)/e)**2));cov=np.linalg.pinv(f.jac.T@f.jac)*max(1,chi/dof)
    info=float(np.linalg.norm(f.jac[:,0]));identifiable=bool(info>1e-4 and abs(f.x[0])<2450 and b[2]>.015)
    return dict(line=name,rest_vacuum=rest,velocity=float(f.x[0]),formal_error=float(np.sqrt(max(cov[0,0],0))),
       gamma=float(f.x[1]),sigma_extra=float(f.x[2]),instrument_sigma=float(inst),depth=float(b[2]),rchi2=chi/dof,
       chi2_gain=chi0-chi,n_pixels=len(w),success=bool(f.success),identifiable=identifiable,
       shape_at_bound=bool(f.x[1]>79 or f.x[2]>34))

def one(row):
    cache=ROOT/'data/helium_line_cache';cache.mkdir(exist_ok=True)
    p=cache/(row['spec_file']+'.json')
    if p.exists():return json.loads(p.read_text())
    records=[]
    with fits.open(ROOT/'raw/helium_spectra'/row['spec_file']) as h:
        for name in REST:
            r=fit_line(h[1].data,name)
            if r:
                r.update(sdss_id=int(row['sdss_id']),spec_file=row['spec_file'],mjd=int(row['mjd']),snr=float(row['snr']))
                records.append(r)
    p.write_text(json.dumps(records,indent=2)+'\n');return records

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all-visits',action='store_true');parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    plan=read_plan();plan=plan[plan.helium_parent&plan.spec_file.map(lambda p:(ROOT/'raw/helium_spectra'/p).exists())]
    if not args.all_visits:plan=plan.sort_values('snr',ascending=False).drop_duplicates('sdss_id')
    print('Currently available visits to fit',len(plan),'stars',plan.sdss_id.nunique(),flush=True)
    rows=[]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(one,plan.to_dict('records')),1):
            rows.extend(result)
            if i%20==0:print('Fitted',i,'/',len(plan),flush=True)
    d=pd.DataFrame(rows)
    if d.empty:return
    d.to_csv(ROOT/'data/helium_initial_line_rvs.csv.gz',index=False)
    good=d[d.success&d.identifiable&(d.chi2_gain>25)&(d.formal_error<100)&~d.shape_at_bound]
    summary=good.groupby('sdss_id').agg(n_lines=('line','size'),median_line_rv=('velocity','median'),line_rv_min=('velocity','min'),line_rv_max=('velocity','max'),median_formal_error=('formal_error','median'),median_rchi2=('rchi2','median')).reset_index()
    summary['absolute_rv_rank']=abs(summary.median_line_rv);summary=summary.sort_values('absolute_rv_rank',ascending=False)
    summary.to_csv(ROOT/'data/helium_absolute_rv_screen.csv',index=False)
    method=dict(n_spectra=len(plan),n_stars=plan.sdss_id.nunique(),laboratory_source='https://physics.nist.gov/PhysRefData/Handbook/Tables/heliumtable2.htm',air_wavelengths=AIR,vacuum_wavelengths=REST,
       caveats='Permitted-component Voigt centroids. Atmospheric pressure/forbidden components, gravitational redshift, calibration and blend errors omitted. Not calibrated systemic velocities; highest ranks require direct inspection. This snapshot only covers downloaded files.')
    (ROOT/'data/helium_initial_rv_method.json').write_text(json.dumps(method,indent=2)+'\n')
    print(summary.head(20).round(2).to_string(index=False),flush=True)

if __name__=='__main__':main()
