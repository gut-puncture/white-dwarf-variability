"""Source attribution and spectra of the selected680-second lead."""
from pathlib import Path
import os,json
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
import numpy as np,pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fast_rotation_tess_validation import fit
ROOT=Path(__file__).resolve().parents[1];SID=109631850

def pixels():
    frozen=json.loads((ROOT/f'data/fast_rotation/{SID}_frozen_discovery_candidates.json').read_text());chosen=frozen['selected'][0]
    f=chosen['frequency'];tref=frozen['tref'];phi=next(p for p in chosen['parameters'] if p['band']=='zr')['phase_radians']+np.pi
    direction=np.array([np.cos(phi),np.sin(phi)])
    paths=sorted((ROOT/f'raw/tess/{SID}').glob('*s_tp.fits'))
    if not paths:raise ValueError(f'No target-pixel files for {SID}')
    fig,axs=plt.subplots(len(paths),3,figsize=(13,4*len(paths)),squeeze=False);records=[]
    for i,path in enumerate(paths):
        with fits.open(path) as h:
            raw=h[1].data;mask=(raw['QUALITY']==0)&np.isfinite(raw['TIME'])&np.all(np.isfinite(raw['FLUX']),axis=(1,2))&np.all(np.isfinite(raw['FLUX_ERR'])&(raw['FLUX_ERR']>0),axis=(1,2))
            t=raw['TIME'][mask].astype(float)+h[1].header['BJDREFI']+h[1].header.get('BJDREFF',0)-2458000
            cube=raw['FLUX'][mask].astype(float);err=raw['FLUX_ERR'][mask].astype(float);sector=int(h[0].header['SECTOR']);ap=h[2].data
            wcs=WCS(h[2].header);x,y=wcs.world_to_pixel_values(h[0].header['RA_OBJ'],h[0].header['DEC_OBJ'])
        mean=np.median(cube,axis=0);yy,xx=np.indices(mean.shape);distance=np.hypot(xx-x,yy-y)
        power=np.zeros_like(mean);signed=np.zeros_like(mean);errors=np.zeros_like(mean);pix=[]
        for py,px in np.ndindex(mean.shape):
            result,_=fit(t,cube[:,py,px],err[:,py,px],f,tref)
            cov=np.array(result['covariance']);inv=np.linalg.inv(cov);c=np.array(result['coefficients'])
            ae=1/np.sqrt(direction@inv@direction);amp=(direction@inv@c)*ae**2
            power[py,px]=result['delta_chi2'];signed[py,px]=amp/ae;errors[py,px]=ae
            pix.append(dict(x=px,y=py,delta_chi2=result['delta_chi2'],fixed_phase_amplitude=amp,fixed_phase_error=ae,fixed_phase_z=amp/ae))
        # Every aperture is defined geometrically or by the archived SPOC mask.
        apertures=[]
        for name,aperture in [('SPOC',(ap&2)>0),('radius1',distance<1),('radius1p5',distance<1.5),('radius2',distance<2)]:
            flux=cube[:,aperture].sum(axis=1);error=np.sqrt((err[:,aperture]**2).sum(axis=1));result,_=fit(t,flux,error,f,tref)
            result.update(name=name,n_pixels=int(aperture.sum()),median_flux=float(np.median(flux)));apertures.append(result)
            pd.DataFrame({'time':t,'flux':flux,'error':error}).to_csv(ROOT/f'data/{SID}_tpf_s{sector}_{name}.csv.gz',index=False)
        # Outer low-flux pixels are a separate background diagnostic, not a
        # variable-star common-mode correction injected into target photometry.
        background=(distance>=3)&(mean<np.quantile(mean[distance>=3],.5))
        bg,_=fit(t,cube[:,background].sum(axis=1),np.sqrt((err[:,background]**2).sum(axis=1)),f,tref)
        k=np.unravel_index(np.argmax(power),power.shape)
        records.append(dict(sector=sector,n=len(t),target_x=float(x),target_y=float(y),apertures=apertures,background=bg,pixels=pix,
            maximum_pixel=dict(x=int(k[1]),y=int(k[0]),delta_chi2=float(power[k]),distance_pixels=float(distance[k]))))
        for j,(arr,title) in enumerate([(np.arcsinh(mean),'Median image (asinh)'),(power,'Fixed-frequency Δχ²'),(signed,'ZTF red phase: signed pixel S/N')]):
            im=axs[i,j].imshow(arr,origin='lower',cmap='RdBu_r' if j==2 else 'viridis');axs[i,j].plot(x,y,'+',color='red' if j<2 else 'black',ms=12,mew=2)
            axs[i,j].contour((ap&2)>0,levels=[.5],colors='white',linewidths=.8);axs[i,j].set(title=f'Sector {sector}: {title}',xlabel='Pixel column',ylabel='Pixel row');fig.colorbar(im,ax=axs[i,j],shrink=.75)
    (ROOT/f'data/{SID}_tess_pixel_checks.json').write_text(json.dumps(dict(frequency=f,tref=tref,expected_flux_phase_from_ztf_red=phi,sectors=records,
        caveat='Each pixel map contains multiple correlated tests; only independently specified target apertures support source attribution. No period-driven aperture optimization.'),indent=2)+'\n')
    fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_tess_pixel_checks.png',dpi=160);plt.close(fig)
    print('Pixels',[(r['sector'],r['maximum_pixel'],[(a['name'],round(a['delta_chi2'],2)) for a in r['apertures']]) for r in records],flush=True)

def spectra():
    fig,axs=plt.subplots(2,1,figsize=(13,8));metadata=[]
    for path in sorted((ROOT/f'raw/fast_rotation/{SID}_spectra').glob('*.fits')):
        with fits.open(path) as h:
            mjd=int(h[0].header['MJD']);d=h[1].data;w=10**d['LOGLAM'];flux=d['FLUX'];iv=d['IVAR'];good=(iv>0)&np.isfinite(flux)&(w>3700)&(w<9000)
            scale=np.median(flux[good&(w>5400)&(w<5700)]);smooth=gaussian_filter1d(flux.astype(float),2)
            axs[0].plot(w[good],smooth[good]/scale,lw=.8,label=f'MJD{mjd}, coadd')
            for k in range(5,len(h)):
                d=h[k].data;f=d['FLUX'];ww=10**d['LOGLAM'];valid=(d['IVAR']>0)&np.isfinite(f)&(ww>3700)&(ww<9000);sc=np.median(f[valid&(ww>5400)&(ww<5700)])
                axs[1].plot(ww[valid],gaussian_filter1d(f.astype(float),3)[valid]/sc,lw=.5,alpha=.65,label=f'{mjd}/{k-5}')
            metadata.append(dict(file=str(path.relative_to(ROOT)),mjd=mjd,total_exposure_seconds=h[0].header.get('EXPTIME'),individual_exposure_seconds=[h[k].header.get('EXPTIME') for k in range(5,len(h))]))
    for ax in axs:
        ax.set(xlabel='Vacuum wavelength (Å)',ylabel='Flux normalized at5400–5700Å',xlim=(3700,9000),ylim=(0,2.3))
        handles,labels=ax.get_legend_handles_labels()
        if len(labels)<=12:ax.legend(ncol=3,fontsize=8)
    axs[0].set_title(f'{SID}: {len(metadata)} SDSS visits across {len(set(r["mjd"] for r in metadata))} nights');axs[1].set_title(f'{SID}: individual SDSS exposures (see metadata for durations)')
    fig.tight_layout();fig.savefig(ROOT/f'figures/{SID}_sdss_spectra.png',dpi=160);plt.close(fig)
    (ROOT/f'data/{SID}_sdss_spectrum_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')

if __name__=='__main__':pixels();spectra()
