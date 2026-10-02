"""Frozen-frequency checks of three independent TESS FFI observing sectors.

The target and a17arcsec brighter neighbor are unresolved by TESS. Apertures
share photons. Background is inferred only from faint pixels outside3pixels.
Daily offsets and slopes, not target-dependent spatial throughput, are fitted.
"""
from pathlib import Path
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord,get_body_barycentric
from astropy.time import Time
from astropy.constants import c
from astropy import units as u
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];SID=61247086
TARGET=SkyCoord(246.24263716223368*u.deg,54.07468268562426*u.deg)
NEIGHBOUR=SkyCoord(246.235956*u.deg,54.072080*u.deg)

def make_nuisance(t,width=1.):
    labels=np.floor((t-t.min())/width).astype(int);unique=np.unique(labels);cols=[]
    for label in unique:
        m=labels==label;cols.append(m.astype(float))
        if sum(m)>=5:cols.append(np.where(m,t-np.mean(t[m]),0.))
    return np.array(cols).T

def fit(t,y,error,f,tref,exptime,B):
    u,s,_=np.linalg.svd(B/error[:,None],full_matrices=False);rank=sum(s>s[0]*1e-10);Q=u[:,:rank]
    def project(a):return a-Q@(Q.T@a)
    wy=project(y/error);phase=2*np.pi*f*(t-tref);X=project(np.sinc(f*exptime/86400)*np.c_[np.sin(phase),np.cos(phase)]/error[:,None])
    coeff=np.linalg.lstsq(X,wy,rcond=None)[0];null=wy@wy;scale=max(1,null/(len(t)-rank));res=wy-X@coeff;gain=float((null-res@res)/scale);cov=np.linalg.inv(X.T@X)*scale
    return dict(delta_chi2=gain,coefficients=coeff.tolist(),covariance=cov.tolist(),amplitude_electrons_per_second=float(np.linalg.norm(coeff)),phase_radians=float(np.arctan2(coeff[1],coeff[0])),variance_inflation=float(scale),n=len(t),nuisance_rank=int(rank),nominal_60_frequency_sector_tail=float(min(1,60*chi2.sf(max(0,gain),2))))

def extract(sector):
    files=list((ROOT/f'raw/tess/{SID}/cutouts').glob(f'tess-s{sector:04d}-*.fits'));assert len(files)==1
    with fits.open(files[0]) as h:
        d=h[1].data;ok=(d['QUALITY']==0)&np.isfinite(d['TIME'])&np.all(np.isfinite(d['FLUX']),axis=(1,2))&np.all(np.isfinite(d['FLUX_ERR'])&(d['FLUX_ERR']>0),axis=(1,2))
        time=np.asarray(d['TIME'][ok],float);oldcorr=np.asarray(d['TIMECORR'][ok],float);flux=np.asarray(d['FLUX'][ok],float);error=np.asarray(d['FLUX_ERR'][ok],float)
        wcs=WCS(h[2].header);x0,y0=wcs.world_to_pixel(TARGET);xn,yn=wcs.world_to_pixel(NEIGHBOUR) if NEIGHBOUR is not None else (np.nan,np.nan);exptime=float(h[1].header['TIMEDEL']*86400)
    spacecraft=time-oldcorr;earth=get_body_barycentric('earth',Time(spacecraft+2457000,format='jd',scale='tdb'))
    corr=np.sum(earth.xyz.to_value(u.au)*TARGET.cartesian.xyz.value[:,None],axis=0)*(1*u.au/c).to_value(u.day);t=spacecraft+corr-1000
    reference=np.median(flux,axis=0);ny,nx=reference.shape;yy,xx=np.indices(reference.shape);distance=np.hypot(xx-x0,yy-y0)
    sky=(distance>3)&(reference<np.percentile(reference,35));design=np.c_[np.ones(nx*ny),(xx-x0).ravel(),(yy-y0).ravel()]
    sigma=np.median(error,axis=0).ravel()[sky.ravel()];matrix=design[sky.ravel()]/sigma[:,None];assert np.linalg.matrix_rank(matrix)==3
    inverse=np.zeros((3,nx*ny));inverse[:,sky.ravel()]=np.linalg.pinv(matrix)/sigma[None,:]
    pixels=flux.reshape(len(t),-1);errors=error.reshape(len(t),-1);background=pixels@inverse.T
    corrected=pixels-background@design.T;frame=pd.DataFrame(dict(time=t,background=background[:,0]))
    for radius in [1.,1.5,2.]:
        aperture=(distance<=radius).ravel().astype(float);operator=aperture-(aperture@design)@inverse;name=f'r{int(radius*100)}'
        frame['flux_'+name]=pixels@operator;frame['error_'+name]=np.sqrt((errors**2)@(operator**2))
    frame.to_csv(ROOT/f'data/{SID}_tesscut_s{sector}_photometry.csv.gz',index=False)
    np.savez_compressed(ROOT/f'data/{SID}_tesscut_s{sector}_pixels.npz',time=t,flux=corrected.astype('f4'),error=errors.astype('f4'),reference=reference,shape=[ny,nx],target_xy=[x0,y0],neighbour_xy=[xn,yn],cadence_seconds=exptime)
    return frame,dict(sector=sector,n_input=len(d),n_clean=len(t),exptime=exptime,target_xy=[float(x0),float(y0)],neighbour_xy=[float(xn),float(yn)] if NEIGHBOUR is not None else None,n_sky_pixels=int(sky.sum()),timing='BJD_TDB−2458000,target-directed geocentric correction; TESS orbital light time omitted (approximately1.5seconds maximum)'),reference

def main():
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());tref=frozen['tref'];results=[];metadata=[]
    fig,axs=plt.subplots(3,2,figsize=(11,11))
    for index,sector in enumerate([16,56,83]):
        d,meta,reference=extract(sector);metadata.append(meta);t=d.time.to_numpy()
        for width in [1.,.5]:
            B=make_nuisance(t,width)
            for radius in [1.,1.5,2.]:
                name=f'r{int(radius*100)}';y=d['flux_'+name].to_numpy();error=d['error_'+name].to_numpy()
                for rank,candidate in enumerate(frozen['selected'],1):
                    r=fit(t,y,error,candidate['frequency'],tref,meta['exptime'],B);r.update(sector=sector,discovery_rank=rank,frequency=candidate['frequency'],aperture_radius=radius,nuisance_segment_days=width);results.append(r)
        primary=next(r for r in results if r['sector']==sector and r['aperture_radius']==1.5 and r['nuisance_segment_days']==1 and r['discovery_rank']==1)
        print('PRIMARY',primary,flush=True)
        ax=axs[index,0];ax.imshow(np.log10(np.maximum(reference,1e-3)),origin='lower',cmap='magma');ax.plot(*meta['target_xy'],'+',color='cyan',ms=12,label='White dwarf');ax.plot(*meta['neighbour_xy'],'x',color='lime',ms=10,label='Bright neighbor');ax.legend(fontsize=8);ax.set(title=f'Sector{sector}: unresolved pair',xlabel='Pixel x',ylabel='Pixel y')
        B=make_nuisance(t);y=d.flux_r150.to_numpy();e=d.error_r150.to_numpy();beta=np.linalg.lstsq(B/e[:,None],y/e,rcond=None)[0];res=y-B@beta;phase=((t-tref)*frozen['selected'][0]['frequency'])%1;ax=axs[index,1]
        for k in range(20):
            m=(phase>=k/20)&(phase<(k+1)/20)
            if sum(m):w=1/e[m]**2;ax.errorbar((k+.5)/20,np.average(res[m],weights=w),np.sqrt(primary['variance_inflation']/sum(w)),fmt='o',color='#37628e')
        ax.set(title=f'Frozen5.55h period: Δχ²={primary["delta_chi2"]:.2f}',xlabel='Phase',ylabel='Combined-aperture residual(e−/s)')
        (ROOT/f'data/{SID}_tesscut_validation.json').write_text(json.dumps(dict(metadata=metadata,tests=results,limitations=[
            'The white dwarf and brighter17arcsecond neighbor are unresolved; a positive aperture signal alone cannot assign the source.',
            'Primary extraction is1.5pixel radius with daily offsets and slopes. Other apertures/nuisance widths are correlated sensitivity checks.',
            'All20frequencies were frozen in ZTF discovery. Three sectors produce60primary frequency/sector tests.',
            'No flux clipping; error inflation estimated under the constant-plus-trend model. Formal chi2 tails need correlated-noise validation if positive.']),indent=2)+'\n')
    fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_tesscut_validation.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
