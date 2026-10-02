"""Measure repeatable line strengths, including hydrogen in the DB cohort.

Fixed wavelength windows and independent sidebands avoid adapting a line
integration window to noise. Formal errors include uncertainty in the fitted
linear continuum, but not resampling covariance or calibration systematics.
Coadds and their exposures are explicitly separate, never independent repeats.
"""
from pathlib import Path
import os,json,concurrent.futures as cf,argparse
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd
from astropy.io import fits
from helium_initial_rv import REST
from helium_cohort_io import read_plan
ROOT=Path(__file__).resolve().parents[1]
LINES=dict(REST,h_alpha=6564.614,h_beta=4862.683)
WINDOWS={'he_4471':(22,32,55),'he_4922':(18,27,45),'he_5876':(18,30,55),'he_6678':(20,32,55),
         'h_alpha':(25,45,75),'h_beta':(25,45,75)}
VERSION=2
# Exposure HDUs can have AND_MASK=0 even for partially rejected extraction
# pixels. The NODATA/BADFLUXFACTOR bits occur throughout valid merged spectra
# and are not used here; reject local extraction/sky/combine failures instead.
REJECT_OR=sum(1<<b for b in [16,18,19,25,27,28])

def measure(d,name):
    w=10**d['loglam'].astype(float);y=d['flux'].astype(float);iv=d['ivar'].astype(float)
    cen=LINES[name];inner,lo,hi=WINDOWS[name];x=(w-cen)/hi;delta=abs(w-cen)
    good=(iv>0)&np.isfinite(y)&(d['and_mask']==0)&((d['or_mask']&REJECT_OR)==0)
    side=good&(delta>lo)&(delta<hi);line=good&(delta<inner)
    # Na D lies just redward of the helium 5876 feature; mask it in this
    # diagnostic so narrow interstellar absorption does not drive its EW.
    if name=='he_5876':line &= ~((w>5888)&(w<5902))
    total=(delta<inner)
    if name=='he_5876':total &= ~((w>5888)&(w<5902))
    if side.sum()<15 or line.sum()<12 or line.sum()<.9*total.sum():return None
    X=np.column_stack([np.ones(side.sum()),x[side]])
    normal=X.T@(iv[side,None]*X)
    if np.linalg.cond(normal)>1e8:return None
    cov=np.linalg.inv(normal);b=cov@(X.T@(iv[side]*y[side]))
    XL=np.column_stack([np.ones(line.sum()),x[line]]);continuum=XL@b
    if np.any(continuum<=0):return None
    dw=np.gradient(w)[line];flux=y[line];ew=float(np.sum(dw*(1-flux/continuum)))
    grad=np.sum((dw*flux/continuum**2)[:,None]*XL,axis=0)
    var=float(np.sum((dw/continuum)**2/iv[line])+grad@cov@grad)
    rc=float(np.sum((y[side]-X@b)**2*iv[side])/(side.sum()-2))
    return dict(line=name,ew=ew,e_ew=np.sqrt(var),continuum=float(b[0]),sideband_rchi2=rc,n_line=int(line.sum()),n_side=int(side.sum()))

def one(row):
    directory=ROOT/'data/helium_strength_cache';directory.mkdir(exist_ok=True)
    cache=directory/(row['spec_file']+'.json')
    if cache.exists():
        old=json.loads(cache.read_text())
        if old.get('version')==VERSION:return old['rows']
    out=[]
    with fits.open(ROOT/'raw/helium_spectra'/row['spec_file']) as h:
        for i,hdu in enumerate(h):
            if i!=1 and not hdu.name.startswith('MJD_EXP_'):continue
            head=h[0].header if i==1 else hdu.header
            if hdu.data is None or 'LOGLAM' not in hdu.data.names:continue
            for name in LINES:
                r=measure(hdu.data,name)
                if r:
                    r.update(sdss_id=int(row['sdss_id']),spec_file=row['spec_file'],hdu=i,
                        level='coadd' if i==1 else 'exposure',mjd=int(row['mjd']),
                        time_mjd_tai=(head.get('TAI-BEG',np.nan)+head.get('TAI-END',np.nan))/(2*86400),
                        exptime=float(head.get('EXPTIME',np.nan)),snr=float(row['snr']))
                    out.append(r)
    cache.write_text(json.dumps(dict(version=VERSION,rows=out),indent=2)+'\n');return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    plan=read_plan()
    plan=plan[plan.helium_parent&plan.spec_file.map(lambda p:(ROOT/'raw/helium_spectra'/p).exists())]
    rows=[]
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(one,plan.to_dict('records')):rows.extend(r)
    d=pd.DataFrame(rows);d.to_csv(ROOT/'data/helium_line_strengths.csv.gz',index=False)
    stats=[]
    for (sid,level,line),g in d.groupby(['sdss_id','level','line']):
        if len(g)<2:continue
        # A 5% of typical EW floor is a ranking stress test, not a calibrated
        # noise model. Changes must later survive flux-level checks.
        error=np.hypot(g.e_ew,.05*abs(np.median(g.ew)))
        wt=1/error**2;mean=float(np.sum(wt*g.ew)/wt.sum());chi=float(np.sum(((g.ew-mean)/error)**2))
        a=g.ew.idxmin();b=g.ew.idxmax();rng=float(g.loc[b,'ew']-g.loc[a,'ew'])
        z=rng/np.hypot(error.loc[a],error.loc[b])
        stats.append(dict(sdss_id=int(sid),level=level,line=line,n=len(g),mean_ew=mean,range_ew=rng,range_z_with_floor=z,
                          chi2_with_floor=chi,reduced_chi2_with_floor=chi/(len(g)-1),n_nights=g.mjd.nunique(),
                          minimum_file=g.loc[a,'spec_file'],minimum_hdu=int(g.loc[a,'hdu']),maximum_file=g.loc[b,'spec_file'],maximum_hdu=int(g.loc[b,'hdu'])))
    s=pd.DataFrame(stats).sort_values('range_z_with_floor',ascending=False);s.to_csv(ROOT/'data/helium_strength_variability.csv',index=False)
    print('Measured',len(d),'line segments from',len(plan),'files;',plan.sdss_id.nunique(),'stars',flush=True)
    print(s[['sdss_id','level','line','n','mean_ew','range_ew','range_z_with_floor']].head(20).round(3).to_string(index=False),flush=True)

if __name__=='__main__':main()
