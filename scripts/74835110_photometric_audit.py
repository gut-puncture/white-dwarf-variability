"""Phase and instrumental-covariate audit of the selected 19-minute lead."""
from pathlib import Path
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from scipy.optimize import minimize_scalar,brentq
from scipy.stats import chi2
from astropy.coordinates import SkyCoord,EarthLocation
from astropy.time import Time
from astropy import units as u
from astropy.utils import iers
iers.conf.auto_download=False
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SELECTED=74.48566828510377
COVARIATES=['airmass','limitmag','magzp','magzprms','clrcoeff','chi','sharp']

def load():
    d=pd.read_csv(ROOT/'data/74835110_rotation_photometry.csv.gz',dtype={'oid':str}).reset_index(drop=True)
    coord=SkyCoord(103.43634989693818*u.deg,1.24722018812986*u.deg)
    site=EarthLocation.from_geodetic(-116.863*u.deg,33.356*u.deg,1706*u.m)
    time=Time(d.mjd.to_numpy(),format='mjd',scale='utc',location=site)
    d['bjd_tdb']=(time.tdb+time.light_travel_time(coord)).jd
    d['time']=d.bjd_tdb-2458000
    return d

def nuisance(d,systematics=False,season=False):
    ids=d.oid.astype(str)
    if season:ids=ids+':'+np.floor((d.mjd-d.mjd.min())/365.25).astype(int).astype(str)
    base=pd.get_dummies(ids,dtype=float).to_numpy();names=list(pd.get_dummies(ids).columns)
    if systematics:
        for band in sorted(d.filtercode.unique()):
            mask=d.filtercode.eq(band).to_numpy()
            for name in COVARIATES:
                v=d[name].to_numpy(float);median=np.nanmedian(v[mask]);scale=np.nanstd(v[mask])
                if scale<1e-9:continue
                value=np.nan_to_num((v-median)/scale);value=np.clip(value,-5,5)*mask
                base=np.c_[base,value];names.append(band+':'+name)
    return base,names

def fit(d,f,systematics=False,jitter=False,season=False):
    x0,names=nuisance(d,systematics,season);bands=sorted(d.filtercode.unique());y=d.mag.to_numpy();e=np.hypot(d.magerr.to_numpy(),.005)
    jit={};p0=np.linalg.lstsq(x0/e[:,None],y/e,rcond=None)[0];rawres=y-x0@p0
    if jitter:
        for b in bands:
            m=d.filtercode.eq(b).to_numpy();denom=max(1,m.sum()-sum(n.startswith(b+':') for n in names)-d.loc[m,'oid'].nunique())
            fun=lambda j:np.sum(rawres[m]**2/(e[m]**2+j*j))-denom
            val=brentq(fun,0,1) if fun(0)>0 else 0.;jit[b]=float(val);e[m]=np.hypot(e[m],val)
        p0=np.linalg.lstsq(x0/e[:,None],y/e,rcond=None)[0]
    phase=2*np.pi*f*d.time.to_numpy();trig=[]
    for b in bands:
        mask=d.filtercode.eq(b).to_numpy();trig.extend([mask*np.sin(phase),mask*np.cos(phase)])
    xs=np.array(trig).T;x=np.c_[x0,xs];coef=np.linalg.lstsq(x/e[:,None],y/e,rcond=None)[0];res=y-x@coef
    q0=float(np.sum(((y-x0@p0)/e)**2));q=float(np.sum((res/e)**2))
    inverse=np.linalg.pinv(x/e[:,None]);cov=inverse@inverse.T;parameters=[]
    for i,b in enumerate(bands):
        ab=coef[len(names)+2*i:len(names)+2*i+2];ca=cov[len(names)+2*i:len(names)+2*i+2,len(names)+2*i:len(names)+2*i+2]
        amplitude=float(np.hypot(*ab));grad=np.array([-ab[1],ab[0]])/amplitude**2
        parameters.append(dict(band=b,amplitude_mag=amplitude,phase_radians=float(np.arctan2(ab[1],ab[0])),phase_error_radians=float(np.sqrt(grad@ca@grad)),coefficients=ab.tolist()))
    # The nuisance projection's response to an injected sinusoid is explicit.
    projected=xs-x0@np.linalg.lstsq(x0/e[:,None],xs/e[:,None],rcond=None)[0]
    responses=[]
    for j in range(xs.shape[1]):responses.append(float(np.sum((projected[:,j]/e)**2)/np.sum((xs[:,j]/e)**2)))
    return dict(n=len(d),frequency=float(f),chi2=q,null_chi2=q0,delta_chi2=q0-q,n_nuisance=len(names),
        n_periodic=len(trig),jitter_mag=jit,parameters=parameters,injection_weighted_power_fraction=responses),dict(residual=res,error=e,null_residual=y-x0@p0,base=x0@coef[:len(names)])

def main():
    d=load();rows=[];step=1/np.ptp(d.time)/5
    for tag,systematics,jitter,season in [('original',False,False,False),('quality_covariates',True,False,False),
        ('quality_and_jitter',True,True,False),('seasons_quality_jitter',True,True,True)]:
        optimize=minimize_scalar(lambda f:fit(d,f,systematics,jitter,season)[0]['chi2'],bounds=(SELECTED-2*step,SELECTED+2*step),method='bounded',options={'xatol':1e-11})
        result,detail=fit(d,optimize.x,systematics,jitter,season);result.update(case=tag,systematics=systematics,jitter=jitter,season_offsets=season)
        rows.append(result);print(tag,json.dumps(result),flush=True)
        if tag=='quality_and_jitter':
            corrected=d.copy();corrected['null_residual']=detail['null_residual'];corrected['adjusted_error']=detail['error'];corrected.to_csv(ROOT/'data/74835110_quality_photometry.csv.gz',index=False)
    splits=[];mid=np.median(d.mjd)
    for name,g in [('early',d[d.mjd<=mid]),('late',d[d.mjd>mid])]+[(b,g) for b,g in d.groupby('filtercode')]:
        result,_=fit(g,rows[2]['frequency'],True,True);result['subset']=name;splits.append(result)
    correlations=[]
    for b,g in d.groupby('filtercode'):
        center=g.mag-g.oid.map(g.groupby('oid').mag.median())
        for col in COVARIATES:correlations.append(dict(band=b,covariate=col,pearson=float(np.corrcoef(center,g[col])[0,1])))
    timing=(d.bjd_tdb-d.hjd)*86400
    out=dict(results=rows,subsets=splits,correlations=correlations,
        timing_bjd_minus_hjd_seconds=np.percentile(timing,[0,50,100]).tolist(),
        exposure_seconds=sorted(d.exptime.unique().tolist()),selected_frequency_hjd=SELECTED,
        caveats=['Selected after 1.99 million frequencies in this target and 47 usable survey targets; no unadjusted single-frequency probability is a discovery claim.',
        'Jitter is fitted conservatively under the nonperiodic model; covariates can remove real signal correlated with observing conditions.',
        'Subsets were not held out from initial selection and are consistency checks, not independent discovery samples.',
        'Gaia identifies a bright variable star6.6arcseconds away, so source attribution remains unresolved.'])
    (ROOT/'data/74835110_photometric_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    fig,axs=plt.subplots(2,3,figsize=(14,8));frequency=rows[2]['frequency']
    for col,(b,g) in enumerate(d.groupby('filtercode')):
        result,detail=fit(g,frequency,True,True);phase=(frequency*g.time.to_numpy())%1;off=detail['base'];y=g.mag.to_numpy()-off;e=detail['error']
        axs[0,col].errorbar(phase,y,e,fmt='.',ms=3,alpha=.4);bins=np.floor(phase*12).astype(int);binned=[]
        for j in range(12):
            mask=bins==j
            if mask.sum():w=1/e[mask]**2;binned.append(((j+.5)/12,np.sum(w*y[mask])/sum(w),1/np.sqrt(sum(w))))
        if binned:bp,by,be=np.array(binned).T;axs[0,col].errorbar(bp,by,be,fmt='o',color='black')
        axs[0,col].invert_yaxis();axs[0,col].set(title=b+' corrected phase',xlabel='Phase',ylabel='Magnitude residual')
        raw=g.mag-g.oid.map(g.groupby('oid').mag.median());axs[1,col].scatter(g['sharp'],raw,s=8,alpha=.5);axs[1,col].set(xlabel='ZTF sharpness',ylabel='Raw magnitude − object median',title=b+' image-shape dependence')
    fig.suptitle('74835110: selected19.3-minute lead; contamination and search trials unresolved')
    fig.tight_layout();fig.savefig(ROOT/'figures/74835110_photometric_audit.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
