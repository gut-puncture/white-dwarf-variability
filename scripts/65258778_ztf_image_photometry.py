"""Deblend difference-image fluxes with archive PSFs and Gaia positions.

Pixel covariance and local PSF mismatch remain limitations; catalogue-level
light curves and these image fits share the same exposures.
"""
import os,json,hashlib,argparse
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']='/tmp/astro-mpl'
from pathlib import Path
import numpy as np,pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[1];SID=65258778
CACHE=ROOT/f'data/{SID}_ztf_image_measurements';CACHE.mkdir(exist_ok=True)
VERSION=4


def readimage(path):
    with fits.open(path) as h:
        z=next(z for z in h if z.data is not None and z.data.ndim==2)
        return z.data.astype(float),z.header.copy()


def design(shape,psf,x,y):
    yy,xx=np.indices(shape);center=(np.array(psf.shape)-1)/2
    return np.array([map_coordinates(psf,[yy-cy+center[0],xx-cx+center[1]],order=3,mode='constant',cval=0.,prefilter=True).ravel() for cx,cy in zip(x,y)]).T


def extract(row,cat):
    key=str(row['filefracday']);outpath=CACHE/f'{key}.json'
    paths={kind:ROOT/f'raw/ztf_images/{SID}/{key}_{kind}.fits' for kind in ['difference','mask','psf']}
    if not all(p.exists() for p in paths.values()):return None
    hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    if outpath.exists():
        old=json.loads(outpath.read_text())
        if old.get('version')==VERSION and old.get('sha256')==hashes:return old
        if old.get('version')==1:outpath.with_suffix('.v1-mask-superseded.json').write_text(json.dumps(old,indent=2)+'\n')
    data,header=readimage(paths['difference']);mask,mh=readimage(paths['mask']);psf,ph=readimage(paths['psf'])
    if data.shape!=mask.shape:raise ValueError('Mask/image shape mismatch')
    if np.any(psf<-.005*np.max(psf)) or not np.isfinite(psf).all() or psf.sum()<=0:raise ValueError('Unexpected PSF')
    psf/=psf.sum();wc=WCS(header);wm=WCS(mh)
    dt=(float(row['mjd'])-57388)/365.25
    ra=cat.ra+cat.pmra.fillna(0)*dt/(3.6e6*np.cos(np.deg2rad(cat.dec)))
    dec=cat.dec+cat.pmdec.fillna(0)*dt/3.6e6
    x,y=wc.all_world2pix(ra,dec,0);xm,ym=wm.all_world2pix(ra,dec,0)
    if np.max(np.hypot(x-xm,y-ym))>1e-5:raise ValueError('Mask/image WCS mismatch')
    templates=design(data.shape,psf,x,y)
    yy,xx=np.indices(data.shape);distance=np.min(np.hypot(xx.ravel()[:,None]-x,yy.ravel()[:,None]-y),axis=1)
    # ZSDS section10.3: bits1 and11 mark valid detected sources, not bad
    # pixels. Official clean-source bit mask is6141. Version1 mask==0
    # incorrectly rejected stellar cores and is archived, never interpreted.
    good=np.isfinite(data.ravel())&((mask.ravel().astype(np.int64)&6141)==0)
    sky=data.ravel()[good&(distance>6)]
    if len(sky)<200:raise ValueError('Insufficient unmasked sky')
    noise=float(1.4826*np.median(abs(sky-np.median(sky))))
    if not noise>0:raise ValueError('Invalid sky noise')
    plane=np.c_[np.ones(data.size),(xx.ravel()-np.mean(xx))/20,(yy.ravel()-np.mean(yy))/20]
    target=int(np.where(cat.source_id.to_numpy()=='1974721783974773376')[0][0])
    close=np.hypot(x-x[target],y-y[target])<8
    # Include first-order positional residuals for all five nearby Gaia
    # sources in a conservative sensitivity model. No centroid phase tuning.
    eps=.05
    dx=(design(data.shape,psf,x[close]+eps,y[close])-design(data.shape,psf,x[close]-eps,y[close]))/(2*eps)
    dy=(design(data.shape,psf,x[close],y[close]+eps)-design(data.shape,psf,x[close],y[close]-eps))/(2*eps)
    injection_ids=['1974721783974773376','1974720310808109696']
    injection_indices=[int(np.where(cat.source_id.to_numpy()==s)[0][0]) for s in injection_ids]
    trials=[]
    for width,offx,offy in [(1.,0.,0.),(1.,.15,0.),(1.,-.15,0.),(1.,0.,.15),(1.,0.,-.15),(.9,0.,0.),(1.1,0.,0.)]:
        yypsf,xxpsf=np.indices(psf.shape);cy,cx=(np.array(psf.shape)-1)/2
        altered=map_coordinates(psf,[(yypsf-cy)/width+cy,(xxpsf-cx)/width+cx],order=3,mode='constant',cval=0.)
        altered/=altered.sum()
        truth=design(data.shape,altered,x[injection_indices]+offx,y[injection_indices]+offy)
        trials.append((dict(width_factor=width,offset_x_pixels=offx,offset_y_pixels=offy),truth))
    rows=[];injection=[];pair_covariances={}
    for mode,extra in [('fixed_positions',plane),('position_derivatives',np.c_[plane,dx,dy])]:
        A=np.c_[templates,extra][good]/noise;obs=data.ravel()[good]/noise
        inverse=np.linalg.pinv(A,rcond=1e-9);beta=inverse@obs;res=obs-A@beta
        rank=np.linalg.matrix_rank(A,tol=1e-9*np.linalg.svd(A,compute_uv=False)[0])
        scale=max(1,float(res@res)/max(1,len(obs)-rank));cov=(inverse@inverse.T)*scale
        factor=10**(-.4*(float(header['MAGZP'])-25.))
        pair_covariances[mode]=(cov[np.ix_(injection_indices,injection_indices)]*factor**2).tolist()
        for k,source in enumerate(cat.source_id):
            coverage=float(np.sum(templates[good,k]**2)/max(np.sum(templates[:,k]**2),1e-30))
            valid=coverage>=.6 and 3<x[k]<data.shape[1]-4 and 3<y[k]<data.shape[0]-4
            rows.append(dict(source_id=source,mode=mode,flux_zp25=float(beta[k]*factor),error_zp25=float(np.sqrt(max(0,cov[k,k]))*factor),coverage=coverage,usable=bool(valid),x=float(x[k]),y=float(y[k])))
        if mode=='fixed_positions':
            # The fitted contribution of a source exactly at each Gaia position
            # is recovered algebraically; this checks linear deblending only,
            # not a mismatched real PSF or imperfect subtraction.
            response=inverse@(templates[good]/noise)
            injection_error=float(np.max(np.abs(response[:len(cat)]-np.eye(len(cat)))))
        for par,truth in trials:
            recovery=inverse@(truth[good]/noise)
            injection.append(dict(mode=mode,**par,source_order=injection_ids,response=recovery[injection_indices].tolist()))
    out=dict(version=VERSION,filefracday=key,sha256=hashes,mjd=float(row['mjd']),time=float(row['time']),split=row['split'],phase=float(row['phase']),noise_pixel=noise,seeing=float(header.get('SEEING',np.nan)),magzp=float(header['MAGZP']),n_good_pixels=int(sum(good)),linear_injection_max_error=injection_error,injection=injection,pair_covariances=pair_covariances,measurements=rows)
    outpath.write_text(json.dumps(out,indent=2)+'\n');return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);args=parser.parse_args()
    plan=pd.read_csv(ROOT/f'data/{SID}_ztf_image_plan.csv',dtype={'filefracday':str})
    if args.limit:plan=plan.iloc[:args.limit]
    cat=pd.read_csv(ROOT/f'raw/{SID}_gaia_neighbours.csv',dtype={'source_id':str})
    rows=[];status=[]
    for i,row in enumerate(plan.to_dict('records')):
        try:out=extract(row,cat)
        except Exception as exc:
            status.append(dict(filefracday=row['filefracday'],status='failed',error=repr(exc)));continue
        if out is None:status.append(dict(filefracday=row['filefracday'],status='not_yet_available'));continue
        status.append(dict(filefracday=row['filefracday'],status='complete',noise_pixel=out['noise_pixel'],linear_injection_max_error=out['linear_injection_max_error']))
        meta={k:out[k] for k in ['filefracday','mjd','time','split','phase','seeing','magzp','noise_pixel']}
        rows.extend(dict(**meta,**r) for r in out['measurements'])
        if (i+1)%20==0:print(i+1,'/',len(plan),'extracted',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/f'data/{SID}_ztf_image_photometry.csv',index=False)
    (ROOT/f'data/{SID}_ztf_image_photometry_status.json').write_text(json.dumps(dict(version=VERSION,results=status),indent=2)+'\n')
    print('DONE',pd.Series([r['status'] for r in status]).value_counts().to_dict(),flush=True)


if __name__=='__main__':main()
